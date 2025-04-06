"""
NLP module for query processing and response generation.
Exports:
    - QueryEngine: Core engine for processing natural language questions
    - ResponseFormatter: Handles formatting of query responses
"""

from .query_engine import QueryEngine
from .formatter import ResponseFormatter

__all__ = ["QueryEngine", "ResponseFormatter"]