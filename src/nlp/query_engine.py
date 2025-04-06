import logging
import numpy as np
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from ..indexing.storage import VectorStorage
from ..indexing.indexer import ContentChunk


@dataclass
class QueryResult:
    """Represents the result of a query."""
    question: str
    answer: str
    source_chunks: List[Tuple[ContentChunk, float]]
    confidence: float
    found_answer: bool


class QueryEngine:
    """Handles processing of natural language questions and retrieval of answers."""

    def __init__(self,
                 vector_store: VectorStorage,
                 embedding_model: Optional[SentenceTransformer] = None,
                 model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 5,
                 confidence_threshold: float = 0.7):
        """
        Initialize the query engine.
        Args:
            vector_store: The vector storage backend
            embedding_model: Pre-initialized embedding model (optional)
            model_name: Name of the embedding model to use if not provided
            top_k: Number of chunks to retrieve for answering
            confidence_threshold: Threshold for determining answer confidence
        """
        self.logger = logging.getLogger(__name__)
        self.vector_store = vector_store
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query by removing unnecessary characters."""
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', '', query)
        return query

    def _embed_query(self, query: str) -> np.ndarray:
        """Convert a query to an embedding vector."""
        return self.embedding_model.encode(query)

    def _compute_answer_confidence(self, chunks: List[Tuple[ContentChunk, float]]) -> float:
        """Compute the confidence score for an answer."""
        if not chunks:
            return 0.0

        avg_similarity = sum(score for _, score in chunks) / len(chunks)
        top_chunk_boost = chunks[0][1] * 0.5
        count_factor = min(len(chunks) / self.top_k, 1.0) * 0.2
        confidence = avg_similarity * 0.3 + top_chunk_boost + count_factor
        return min(max(confidence, 0.0), 1.0)

    def _generate_answer(self, query: str, relevant_chunks: List[Tuple[ContentChunk, float]]) -> str:
        """Generate an answer from relevant chunks."""
        if not relevant_chunks:
            return "I couldn't find any relevant information in the documentation."
        best_chunk = relevant_chunks[0][0]
        return best_chunk.text

    def process_query(self, query: str) -> QueryResult:
        """Process a natural language query and retrieve an answer."""
        query = self._preprocess_query(query)

        if not query:
            return QueryResult(
                question=query,
                answer="Your query is empty. Please provide a valid question.",
                source_chunks=[],
                confidence=0.0,
                found_answer=False
            )

        self.logger.info(f"Processing query: {query}")
        query_vector = self._embed_query(query)
        relevant_chunks = self.vector_store.search(query_vector, k=self.top_k)
        confidence = self._compute_answer_confidence(relevant_chunks)
        answer = self._generate_answer(query, relevant_chunks)
        found_answer = confidence >= self.confidence_threshold

        if not found_answer:
            answer = "I don't have enough information to provide a confident answer."

        return QueryResult(
            question=query,
            answer=answer,
            source_chunks=relevant_chunks,
            confidence=confidence,
            found_answer=found_answer
        )