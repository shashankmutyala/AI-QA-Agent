import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from src.crawling.parser import ParsedContent

logger = logging.getLogger(__name__)


class ContentChunk:
    """A chunk of content with its embedding."""

    def __init__(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Initialize a content chunk."""
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}


class Indexer:
    """Process documents and generate embeddings."""

    def __init__(self, config):
        """Initialize the indexer with configuration."""
        self.chunk_size = getattr(config, 'chunk_size', 512)
        self.chunk_overlap = getattr(config, 'chunk_overlap', 128)
        self.min_chunk_size = getattr(config, 'min_chunk_size', 50)
        self.vector_dimension = getattr(config, 'vector_dimension', 384)
        self.batch_size = getattr(config, 'batch_size', 32)

        # Use local model path instead of downloading
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                        "downloaded_model")

        logger.info(f"Loading embedding model from local path: {local_model_path}")
        try:
            self.embedding_model = SentenceTransformer(local_model_path)
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            model_name = getattr(config, 'embedding_model', 'all-MiniLM-L6-v2')
            logger.info(f"Falling back to loading from model name: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for text."""
        return self.embedding_model.encode(text, show_progress_bar=False)

    def _create_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[ContentChunk]:
        """Split text into chunks with overlap."""
        if not text or len(text) < self.min_chunk_size:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Don't create tiny chunks at the end
            if end - start < self.min_chunk_size:
                break

            chunk_text = text[start:end]
            embedding = self.get_embedding(chunk_text)

            chunk = ContentChunk(
                text=chunk_text,
                embedding=embedding,
                metadata=metadata.copy() if metadata else {}
            )
            chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def process_documents(self, documents: List[ParsedContent]) -> List[ContentChunk]:
        """Process documents to create content chunks with embeddings."""
        all_chunks = []

        for doc in documents:
            # Add basic document metadata
            metadata = {
                "url": doc.url,
                "title": doc.title,
            }

            # Process main text
            if doc.text:
                all_chunks.extend(self._create_chunks(doc.text, metadata))

            # Process each section individually for better context
            for section in doc.sections:
                section_metadata = metadata.copy()
                section_metadata["section_title"] = section["title"]

                all_chunks.extend(self._create_chunks(section["content"], section_metadata))

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def batch_process(self, texts: List[str], batch_size: int = None) -> List[np.ndarray]:
        """Process a batch of texts to get embeddings."""
        batch_size = batch_size or self.batch_size
        return self.embedding_model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    def merge_small_chunks(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return []

        result = []
        current_text = ""
        current_metadata = None

        for chunk in chunks:
            # Start a new merged chunk
            if not current_text:
                current_text = chunk.text
                current_metadata = chunk.metadata.copy()
                continue

            # If combining would make it too large, finalize current chunk
            if len(current_text) + len(chunk.text) > self.chunk_size:
                embedding = self.get_embedding(current_text)
                result.append(ContentChunk(
                    text=current_text,
                    embedding=embedding,
                    metadata=current_metadata
                ))
                current_text = chunk.text
                current_metadata = chunk.metadata.copy()
            else:
                # Merge with current chunk
                current_text += " " + chunk.text

        # Add the last chunk if there is one
        if current_text:
            embedding = self.get_embedding(current_text)
            result.append(ContentChunk(
                text=current_text,
                embedding=embedding,
                metadata=current_metadata
            ))

        return result

    def chunk_by_section(self, document: ParsedContent) -> List[ContentChunk]:
        """Create chunks based on sections of the document."""
        chunks = []

        # Basic document metadata
        metadata = {
            "url": document.url,
            "title": document.title,
        }

        # Process each section individually
        for section in document.sections:
            if not section["content"] or len(section["content"]) < self.min_chunk_size:
                continue

            section_metadata = metadata.copy()
            section_metadata["section_title"] = section["title"]

            # Split section content into chunks if needed
            section_chunks = self._create_chunks(section["content"], section_metadata)

            # If section is small enough, keep as one chunk
            if not section_chunks and len(section["content"]) >= self.min_chunk_size:
                embedding = self.get_embedding(section["content"])
                chunks.append(ContentChunk(
                    text=section["content"],
                    embedding=embedding,
                    metadata=section_metadata
                ))
            else:
                chunks.extend(section_chunks)

        # If no sections or all sections were too small, process the full text
        if not chunks and document.text and len(document.text) >= self.min_chunk_size:
            chunks.extend(self._create_chunks(document.text, metadata))

        return chunks