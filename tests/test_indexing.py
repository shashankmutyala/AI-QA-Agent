import os
import sys
import pytest
import shutil
import tempfile
import numpy as np
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crawling.parser import ParsedContent
from src.indexing.indexer import Indexer, ContentChunk
from src.indexing.storage import FAISSStorage


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for index storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_parsed_content():
    """Create sample parsed content for testing."""
    return [
        ParsedContent(
            url="https://example.com/page1",
            title="Test Page 1",
            text="This is a test page about indexing content. It contains information about how the system works.",
            sections=[
                {
                    "title": "Introduction",
                    "content": "This is a test page about indexing content."
                },
                {
                    "title": "Details",
                    "content": "It contains information about how the system works."
                }
            ],
            metadata={"language": "en"}
        ),
        ParsedContent(
            url="https://example.com/page2",
            title="Test Page 2",
            text="This is another page with different content. It explains the search functionality.",
            sections=[
                {
                    "title": "Search",
                    "content": "This is another page with different content. It explains the search functionality."
                }
            ],
            metadata={"language": "en"}
        )
    ]


@pytest.fixture
def indexer_config():
    """Create an indexer configuration."""
    return {
        "chunk_size": 100,
        "chunk_overlap": 20,
        "embedding_model": "all-MiniLM-L6-v2",
        "min_chunk_size": 10,
        "vector_dimension": 384,
        "batch_size": 32
    }


@pytest.fixture
def indexer(indexer_config):
    """Create an indexer for testing."""
    return Indexer(indexer_config)


def test_indexer_initialization(indexer):
    """Test that the indexer initializes correctly."""
    assert indexer.chunk_size == 100
    assert indexer.chunk_overlap == 20
    assert indexer.min_chunk_size == 10


def test_process_documents(indexer, sample_parsed_content):
    """Test processing documents into chunks with embeddings."""
    chunks = indexer.process_documents(sample_parsed_content)

    # Verify chunks were created
    assert len(chunks) > 0

    # Check that each chunk has the expected properties
    for chunk in chunks:
        assert isinstance(chunk, ContentChunk)
        assert chunk.text
        assert chunk.embedding is not None
        assert len(chunk.embedding) == indexer.vector_dimension
        assert isinstance(chunk.metadata, dict)
        assert "url" in chunk.metadata
        assert "title" in chunk.metadata


def test_chunk_creation_with_sections(indexer, sample_parsed_content):
    """Test that chunks are created from document sections."""
    chunks = indexer.process_documents(sample_parsed_content)

    # Check that section information is preserved in chunks
    section_chunks = [c for c in chunks if "section_title" in c.metadata]
    assert len(section_chunks) > 0

    # Verify section titles are captured in metadata
    section_titles = [c.metadata.get("section_title") for c in section_chunks]
    assert "Introduction" in section_titles or "Details" in section_titles or "Search" in section_titles


def test_minimum_chunk_size(indexer, sample_parsed_content):
    """Test that chunks smaller than min_chunk_size are handled properly."""
    # Change config to have a larger minimum chunk size
    indexer.min_chunk_size = 500

    # This should result in fewer chunks due to the high minimum size
    chunks = indexer.process_documents(sample_parsed_content)

    # Small sections should be combined or skipped
    for chunk in chunks:
        assert len(chunk.text) >= indexer.min_chunk_size


def test_storage_initialization():
    """Test initializing the vector store."""
    dimension = 384
    storage = FAISSStorage(dimension=dimension)

    assert storage.dimension == dimension
    assert len(storage.chunks) == 0


def test_storage_add_chunks(indexer, sample_parsed_content):
    """Test adding chunks to the vector store."""
    chunks = indexer.process_documents(sample_parsed_content)

    storage = FAISSStorage(dimension=indexer.vector_dimension)
    storage.add_chunks(chunks)

    assert len(storage.chunks) == len(chunks)
    assert storage.index is not None


def test_storage_save_load(indexer, sample_parsed_content, temp_storage_dir):
    """Test saving and loading the vector store."""
    chunks = indexer.process_documents(sample_parsed_content)

    storage = FAISSStorage(dimension=indexer.vector_dimension)
    storage.add_chunks(chunks)

    index_path = os.path.join(temp_storage_dir, "test_index")
    storage.save(index_path)

    assert os.path.exists(index_path)

    # Load into a new storage instance
    new_storage = FAISSStorage(dimension=indexer.vector_dimension)
    new_storage.load(index_path)

    assert len(new_storage.chunks) == len(chunks)

    # Check that we can query the loaded index
    query_embedding = np.random.rand(indexer.vector_dimension).astype(np.float32)
    results = new_storage.similarity_search(query_embedding, k=1)

    assert len(results) == 1


def test_similarity_search(indexer, sample_parsed_content):
    """Test similarity search functionality."""
    chunks = indexer.process_documents(sample_parsed_content)

    storage = FAISSStorage(dimension=indexer.vector_dimension)
    storage.add_chunks(chunks)

    # Use the embedding from one of the chunks as a query
    query_chunk = chunks[0]
    query_embedding = query_chunk.embedding

    results = storage.similarity_search(query_embedding, k=2)

    # Should get at least one result
    assert len(results) > 0

    # First result should be the original chunk (highest similarity)
    first_result_chunk, first_result_score = results[0]
    assert first_result_chunk.text == query_chunk.text
    assert first_result_score > 0.9  # Should have very high similarity to itself


def test_incremental_updates(indexer, sample_parsed_content):
    """Test adding new chunks to an existing index."""
    # Split the sample content
    first_content = [sample_parsed_content[0]]
    second_content = [sample_parsed_content[1]]

    # Process and add the first content
    first_chunks = indexer.process_documents(first_content)
    storage = FAISSStorage(dimension=indexer.vector_dimension)
    storage.add_chunks(first_chunks)

    initial_chunk_count = len(storage.chunks)

    # Process and add the second content
    second_chunks = indexer.process_documents(second_content)
    storage.add_chunks(second_chunks)

    # Verify the chunks were added
    assert len(storage.chunks) == initial_chunk_count + len(second_chunks)

    # Verify we can search across all chunks
    query_embedding = indexer.get_embedding("search functionality")
    results = storage.similarity_search(query_embedding, k=2)

    # Should find content from the second page
    assert any("search" in result[0].text.lower() for result in results)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])