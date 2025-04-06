"""
Storage module for vector storage implementations.
Exports:
    - VectorStorage: Abstract base class for vector storage backends
    - FAISSStorage: FAISS-based implementation for vector similarity search
"""

from .storage import VectorStorage, FAISSStorage

__all__ = ["VectorStorage", "FAISSStorage"]