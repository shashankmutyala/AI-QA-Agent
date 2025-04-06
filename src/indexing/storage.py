import os
import logging
import pickle
import numpy as np
import faiss
from typing import List, Tuple
from abc import ABC, abstractmethod
from .indexer import ContentChunk


class VectorStorage(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add_chunks(self, chunks: List[ContentChunk]) -> None:
        """Add content chunks to the storage."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[ContentChunk, float]]:
        """Search for similar chunks using vector similarity."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector storage to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector storage from disk."""
        pass


class FAISSStorage(VectorStorage):
    """FAISS-based vector storage implementation."""

    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS storage.

        Args:
            dimension: Dimension of the embedding vectors
        """
        self.logger = logging.getLogger(__name__)
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize the FAISS index."""
        self.logger.info(f"Initializing FAISS index with dimension {self.dimension}")
        self.index = faiss.IndexFlatL2(self.dimension)

    def _validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """Validate that embeddings have the correct dimensionality."""
        if embeddings.shape[1] != self.dimension:
            self.logger.error(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}.")
            return False
        return True

    def add_chunks(self, chunks: List[ContentChunk]) -> None:
        """Add content chunks to the FAISS index."""
        if not chunks:
            self.logger.warning("No chunks provided to add to the index")
            return

        embeddings = np.array([chunk.embedding for chunk in chunks if chunk.embedding is not None])
        valid_chunks = [chunk for chunk in chunks if chunk.embedding is not None]

        if embeddings.size == 0 or not self._validate_embeddings(embeddings):
            self.logger.warning("No valid embeddings found or embedding dimension mismatch")
            return

        start_idx = len(self.chunks)
        self.logger.info(f"Adding {len(valid_chunks)} chunks to FAISS index")
        self.index.add(embeddings)
        self.chunks.extend(valid_chunks)
        self.logger.info(f"Index now contains {len(self.chunks)} chunks")

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[ContentChunk, float]]:
        """Search for similar chunks using vector similarity."""
        if len(self.chunks) == 0:
            self.logger.warning("Search called on empty index")
            return []

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        if not self._validate_embeddings(query_vector):
            raise ValueError("Query vector dimensionality mismatch")

        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                similarity = 1.0 / (1.0 + distances[0][i])
                results.append((self.chunks[idx], similarity))

        return results

    def save(self, path: str) -> None:
        """Save the FAISS index and chunks to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))

        with open(os.path.join(path, "chunks.pkl"), 'wb') as f:
            pickle.dump([chunk for chunk in self.chunks], f)

    def load(self, path: str) -> None:
        """Load the FAISS index and chunks from disk."""
        self.index = faiss.read_index(os.path.join(path, "faiss_index.bin"))
        with open(os.path.join(path, "chunks.pkl"), 'rb') as f:
            self.chunks = pickle.load(f)
        self.logger.info(f"Loaded index with {len(self.chunks)} chunks")