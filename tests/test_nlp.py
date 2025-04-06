import os
import sys
import pytest
from unittest import mock
import numpy as np
from typing import List, Tuple, Dict, Any

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.indexer import ContentChunk
from src.indexing.storage import FAISSStorage
from src.nlp.query_engine import QueryEngine, QueryResult
from src.nlp.formatter import ResponseFormatter


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self.chunks = []
        self.dimension = 384
        self.search_called = False
        self.last_query = None
        self.last_k = None

    def add_chunks(self, chunks: List[ContentChunk]):
        """Add chunks to the mock store."""
        self.chunks.extend(chunks)

    def similarity_search(self, query_vector: np.ndarray, k: int) -> List[Tuple[ContentChunk, float]]:
        """Mock similarity search."""
        self.search_called = True
        self.last_query = query_vector
        self.last_k = k

        # Return some mock results with decreasing scores
        results = []
        for i, chunk in enumerate(self.chunks[:k]):
            # Score decreases with each result
            score = 0.9 - (0.1 * i)
            results.append((chunk, score))

        return results


@pytest.fixture
def sample_chunks():
    """Create sample content chunks for testing."""
    return [
        ContentChunk(
            text="To reset your password, click on Forgot Password and follow the instructions sent to your email.",
            embedding=np.random.rand(384).astype(np.float32),
            metadata={
                "url": "https://example.com/help/passwords",
                "title": "Password Reset Guide",
                "section_title": "Reset Steps"
            }
        ),
        ContentChunk(
            text="Account settings can be found in the profile menu. You can change your email, name and notification preferences.",
            embedding=np.random.rand(384).astype(np.float32),
            metadata={
                "url": "https://example.com/help/account",
                "title": "Account Settings",
                "section_title": "Profile Settings"
            }
        ),
        ContentChunk(
            text="Two-factor authentication adds an extra layer of security to your account. You can enable it in security settings.",
            embedding=np.random.rand(384).astype(np.float32),
            metadata={
                "url": "https://example.com/help/security",
                "title": "Security Features",
                "section_title": "Two-Factor Authentication"
            }
        )
    ]


@pytest.fixture
def mock_vector_store(sample_chunks):
    """Create a mock vector store with sample chunks."""
    store = MockVectorStore()
    store.add_chunks(sample_chunks)
    return store


@pytest.fixture
def query_engine(mock_vector_store):
    """Create a query engine with mock vector store."""
    return QueryEngine(
        vector_store=mock_vector_store,
        top_k=3,
        confidence_threshold=0.6
    )


def test_query_engine_initialization(mock_vector_store):
    """Test that the query engine initializes correctly."""
    engine = QueryEngine(
        vector_store=mock_vector_store,
        top_k=5,
        confidence_threshold=0.7
    )

    assert engine.top_k == 5
    assert engine.confidence_threshold == 0.7
    assert engine.vector_store == mock_vector_store


def test_process_query(query_engine, mock_vector_store):
    """Test processing a query."""
    with mock.patch.object(query_engine, 'get_embedding', return_value=np.random.rand(384).astype(np.float32)):
        result = query_engine.process_query("How do I reset my password?")

    # Check that vector store was queried
    assert mock_vector_store.search_called
    assert mock_vector_store.last_k == query_engine.top_k

    # Check result structure
    assert isinstance(result, QueryResult)
    assert result.query == "How do I reset my password?"
    assert result.answer is not None
    assert len(result.source_chunks) > 0
    assert result.confidence > 0


def test_query_without_matches(query_engine, mock_vector_store):
    """Test behavior when no good matches are found."""
    # Set a very high confidence threshold
    query_engine.confidence_threshold = 0.99

    with mock.patch.object(query_engine, 'get_embedding', return_value=np.random.rand(384).astype(np.float32)):
        result = query_engine.process_query("Something completely unrelated")

    # Should indicate low confidence
    assert result.confidence < query_engine.confidence_threshold
    assert "insufficient information" in result.answer.lower() or "don't have enough" in result.answer.lower()


def test_answer_generation_with_relevant_context(query_engine):
    """Test answer generation with relevant context."""
    # Override the get_relevant_chunks method to return a specific chunk
    relevant_chunk = ContentChunk(
        text="To reset your password, click on Forgot Password and follow the instructions sent to your email.",
        embedding=np.random.rand(384).astype(np.float32),
        metadata={
            "url": "https://example.com/help/passwords",
            "title": "Password Reset Guide"
        }
    )

    with mock.patch.object(query_engine, '_get_relevant_chunks', return_value=[(relevant_chunk, 0.95)]):
        result = query_engine.process_query("How do I reset my password?")

    # Check that the answer contains information from the relevant chunk
    assert "forgot password" in result.answer.lower() or "reset" in result.answer.lower()
    assert result.confidence > query_engine.confidence_threshold


def test_response_formatter_initialization():
    """Test response formatter initialization."""
    formatter = ResponseFormatter(
        show_sources=True,
        show_confidence=True,
        max_sources=2,
        min_confidence_for_source=0.7
    )

    assert formatter.show_sources is True
    assert formatter.show_confidence is True
    assert formatter.max_sources == 2
    assert formatter.min_confidence_for_source == 0.7


def test_format_result_plain_text(query_engine):
    """Test formatting results as plain text."""
    # Create a sample query result
    result = QueryResult(
        query="How do I reset my password?",
        answer="To reset your password, click on Forgot Password and follow the instructions sent to your email.",
        source_chunks=[
            (ContentChunk(
                text="To reset your password, click on Forgot Password and follow the instructions sent to your email.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/passwords",
                    "title": "Password Reset Guide"
                }
            ), 0.95)
        ],
        confidence=0.95
    )

    formatter = ResponseFormatter(show_sources=True, show_confidence=True)
    output = formatter.format_result(result, format_type="plain")

    # Check that output contains expected elements
    assert "reset your password" in output
    assert "Confidence: 0.95" in output
    assert "https://example.com/help/passwords" in output


def test_format_result_markdown(query_engine):
    """Test formatting results as markdown."""
    # Create a sample query result
    result = QueryResult(
        query="How do I reset my password?",
        answer="To reset your password, click on Forgot Password and follow the instructions sent to your email.",
        source_chunks=[
            (ContentChunk(
                text="To reset your password, click on Forgot Password and follow the instructions sent to your email.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/passwords",
                    "title": "Password Reset Guide"
                }
            ), 0.95)
        ],
        confidence=0.95
    )

    formatter = ResponseFormatter(show_sources=True, show_confidence=True)
    output = formatter.format_result(result, format_type="markdown")

    # Check that output contains expected markdown elements
    assert "## Answer" in output
    assert "reset your password" in output
    assert "## Sources" in output
    assert "[Password Reset Guide](https://example.com/help/passwords)" in output


def test_format_without_sources(query_engine):
    """Test formatting without showing sources."""
    result = QueryResult(
        query="How do I reset my password?",
        answer="To reset your password, click on Forgot Password and follow the instructions sent to your email.",
        source_chunks=[
            (ContentChunk(
                text="To reset your password, click on Forgot Password and follow the instructions sent to your email.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/passwords",
                    "title": "Password Reset Guide"
                }
            ), 0.95)
        ],
        confidence=0.95
    )

    formatter = ResponseFormatter(show_sources=False, show_confidence=True)
    output = formatter.format_result(result, format_type="plain")

    # Check that output does not contain sources
    assert "Sources:" not in output
    assert "https://example.com/help/passwords" not in output
    assert "To reset your password" in output


def test_format_max_sources_limit(query_engine):
    """Test limiting the number of sources shown."""
    result = QueryResult(
        query="What security features are available?",
        answer="You can enable two-factor authentication and set up security questions.",
        source_chunks=[
            (ContentChunk(
                text="Two-factor authentication adds an extra layer of security.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/security/2fa",
                    "title": "Two-Factor Authentication"
                }
            ), 0.95),
            (ContentChunk(
                text="Security questions help verify your identity when recovering accounts.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/security/questions",
                    "title": "Security Questions"
                }
            ), 0.85),
            (ContentChunk(
                text="Password policies ensure your password is strong enough.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/security/passwords",
                    "title": "Password Policies"
                }
            ), 0.75)
        ],
        confidence=0.85
    )

    formatter = ResponseFormatter(show_sources=True, show_confidence=True, max_sources=2)
    output = formatter.format_result(result, format_type="plain")

    # Should only show the top 2 sources
    assert "Two-Factor Authentication" in output
    assert "Security Questions" in output
    assert "Password Policies" not in output


def test_format_confidence_threshold(query_engine):
    """Test filtering sources by confidence threshold."""
    result = QueryResult(
        query="What security features are available?",
        answer="You can enable two-factor authentication and set up security questions.",
        source_chunks=[
            (ContentChunk(
                text="Two-factor authentication adds an extra layer of security.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/security/2fa",
                    "title": "Two-Factor Authentication"
                }
            ), 0.95),
            (ContentChunk(
                text="Security questions help verify your identity when recovering accounts.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/security/questions",
                    "title": "Security Questions"
                }
            ), 0.85),
            (ContentChunk(
                text="Password policies ensure your password is strong enough.",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={
                    "url": "https://example.com/help/security/passwords",
                    "title": "Password Policies"
                }
            ), 0.65)
        ],
        confidence=0.85
    )

    formatter = ResponseFormatter(show_sources=True, show_confidence=True, min_confidence_for_source=0.8)
    output = formatter.format_result(result, format_type="plain")

    # Should only show sources with confidence >= 0.8
    assert "Two-Factor Authentication" in output
    assert "Security Questions" in output
    assert "Password Policies" not in output


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])