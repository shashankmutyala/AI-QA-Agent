import os
import sys
import pytest
import shutil
import tempfile
import responses
from pathlib import Path
from unittest import mock

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, AppConfig
from src.crawling.crawler import Crawler
from src.crawling.parser import Parser
from src.indexing.indexer import Indexer
from src.indexing.storage import FAISSStorage
from src.nlp.query_engine import QueryEngine
from src.nlp.formatter import ResponseFormatter

# Sample website content for end-to-end testing
MAIN_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Help Center</title>
</head>
<body>
    <header>
        <h1>Welcome to the Help Center</h1>
    </header>
    <main>
        <p>This is the main help page. Please find links to common topics below:</p>
        <ul>
            <li><a href="/password-reset">How to reset your password</a></li>
            <li><a href="/account-settings">Account settings</a></li>
        </ul>
    </main>
</body>
</html>
"""

PASSWORD_RESET_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Password Reset Guide</title>
</head>
<body>
    <header>
        <h1>How to Reset Your Password</h1>
    </header>
    <main>
        <section>
            <h2>Password Reset Steps</h2>
            <p>To reset your password, follow these steps:</p>
            <ol>
                <li>Click on "Forgot Password" on the login page</li>
                <li>Enter your email address</li>
                <li>Check your email for a reset link</li>
                <li>Click the link and enter your new password</li>
                <li>Submit the form to save your new password</li>
            </ol>
        </section>
        <section>
            <h2>Troubleshooting</h2>
            <p>If you don't receive the reset email:</p>
            <ul>
                <li>Check your spam folder</li>
                <li>Verify you used the correct email address</li>
                <li>Wait a few minutes and try again</li>
            </ul>
        </section>
    </main>
</body>
</html>
"""

ACCOUNT_SETTINGS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Account Settings</title>
</head>
<body>
    <header>
        <h1>Account Settings</h1>
    </header>
    <main>
        <section>
            <h2>Managing Your Profile</h2>
            <p>You can update your profile information by going to the Settings page.</p>
            <p>Available profile settings include:</p>
            <ul>
                <li>Name and contact information</li>
                <li>Profile picture</li>
                <li>Email preferences</li>
            </ul>
        </section>
        <section>
            <h2>Privacy Settings</h2>
            <p>Control your privacy settings by clicking on the Privacy tab in your account settings.</p>
            <p>You can adjust who can see your profile and activities.</p>
        </section>
    </main>
</body>
</html>
"""


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for index storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config(temp_storage_dir):
    """Create a test configuration."""
    config = load_config()
    config.storage.storage_dir = temp_storage_dir
    config.crawler.max_depth = 2
    config.crawler.max_pages = 10
    config.crawler.rate_limit = 0.0  # No delay in tests
    return config


@responses.activate
def test_end_to_end_workflow(mock_config, temp_storage_dir):
    """Test the complete workflow from crawling to querying."""
    # Set up mock responses
    base_url = "https://test-help-center.com"

    responses.add(
        responses.GET,
        base_url,
        body=MAIN_PAGE_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/password-reset",
        body=PASSWORD_RESET_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/account-settings",
        body=ACCOUNT_SETTINGS_HTML,
        status=200,
        content_type="text/html"
    )

    # Step 1: Crawl the website
    crawler = Crawler(mock_config.crawler)
    pages = crawler.crawl(base_url)

    assert len(pages) == 3
    assert base_url in pages
    assert f"{base_url}/password-reset" in pages
    assert f"{base_url}/account-settings" in pages

    # Step 2: Parse the content
    parser = Parser()
    parsed_contents = []

    for url, html in pages.items():
        parsed_content = parser.parse_page(url, html)
        parsed_contents.append(parsed_content)

    assert len(parsed_contents) == 3

    # Step 3: Generate embeddings and index
    indexer = Indexer(mock_config.indexing)
    chunks = indexer.process_documents(parsed_contents)

    assert len(chunks) > 0

    # Step 4: Store in vector database
    storage = FAISSStorage(dimension=mock_config.indexing.vector_dimension)
    storage.add_chunks(chunks)

    # Save the index
    domain = base_url.replace('http://', '').replace('https://', '').split('/')[0]
    index_dir = Path(temp_storage_dir) / f"index_{domain.replace('.', '_')}"
    storage.save(str(index_dir))

    assert index_dir.exists()

    # Step 5: Query the index
    query_engine = QueryEngine(
        vector_store=storage,
        top_k=mock_config.query.top_k,
        confidence_threshold=mock_config.query.confidence_threshold
    )

    # Test password reset query
    password_result = query_engine.process_query("How do I reset my password?")
    assert password_result.answer is not None
    assert password_result.confidence > 0.0
    assert len(password_result.source_chunks) > 0
    # Check that the answer contains key information
    password_keywords = ["forgot", "password", "email", "reset", "link"]
    assert any(keyword.lower() in password_result.answer.lower() for keyword in password_keywords)

    # Test account settings query
    settings_result = query_engine.process_query("What can I do in account settings?")
    assert settings_result.answer is not None
    assert settings_result.confidence > 0.0
    assert len(settings_result.source_chunks) > 0
    # Check that the answer contains key information
    settings_keywords = ["profile", "privacy", "settings", "information"]
    assert any(keyword.lower() in settings_result.answer.lower() for keyword in settings_keywords)

    # Step 6: Format results
    formatter = ResponseFormatter(
        show_sources=mock_config.output.show_sources,
        show_confidence=mock_config.output.show_confidence
    )

    # Test markdown formatting
    markdown_output = formatter.format_result(password_result, format_type="markdown")
    assert markdown_output is not None
    assert "password" in markdown_output.lower()
    assert "Sources:" in markdown_output

    # Test plain formatting
    plain_output = formatter.format_result(settings_result, format_type="plain")
    assert plain_output is not None
    assert "settings" in plain_output.lower()


@responses.activate
def test_incremental_indexing(mock_config, temp_storage_dir):
    """Test that new pages can be added to an existing index."""
    # Step 1: Crawl and index initial pages
    base_url = "https://test-help-center.com"

    responses.add(
        responses.GET,
        base_url,
        body=MAIN_PAGE_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/password-reset",
        body=PASSWORD_RESET_HTML,
        status=200,
        content_type="text/html"
    )

    # Initial crawl and indexing
    crawler = Crawler(mock_config.crawler)
    pages = crawler.crawl(base_url)

    parser = Parser()
    parsed_contents = [parser.parse_page(url, html) for url, html in pages.items()]

    indexer = Indexer(mock_config.indexing)
    chunks = indexer.process_documents(parsed_contents)

    storage = FAISSStorage(dimension=mock_config.indexing.vector_dimension)
    storage.add_chunks(chunks)

    initial_chunk_count = len(storage.chunks)

    # Step 2: Add a new page to the index
    responses.add(
        responses.GET,
        f"{base_url}/account-settings",
        body=ACCOUNT_SETTINGS_HTML,
        status=200,
        content_type="text/html"
    )

    new_pages = {f"{base_url}/account-settings": ACCOUNT_SETTINGS_HTML}
    new_parsed = [parser.parse_page(url, html) for url, html in new_pages.items()]
    new_chunks = indexer.process_documents(new_parsed)

    storage.add_chunks(new_chunks)

    # Verify new chunks were added
    assert len(storage.chunks) > initial_chunk_count

    # Step 3: Query for the new content
    query_engine = QueryEngine(
        vector_store=storage,
        top_k=mock_config.query.top_k,
        confidence_threshold=mock_config.query.confidence_threshold
    )

    result = query_engine.process_query("What privacy settings are available?")

    assert result.answer is not None
    assert "privacy" in result.answer.lower()


@responses.activate
def test_error_recovery(mock_config, temp_storage_dir):
    """Test system recovery from crawl errors."""
    base_url = "https://test-help-center.com"

    # Set up responses with one error
    responses.add(
        responses.GET,
        base_url,
        body=MAIN_PAGE_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/password-reset",
        status=500  # Error response
    )

    responses.add(
        responses.GET,
        f"{base_url}/account-settings",
        body=ACCOUNT_SETTINGS_HTML,
        status=200,
        content_type="text/html"
    )

    # Crawl should continue despite the error
    crawler = Crawler(mock_config.crawler)
    pages = crawler.crawl(base_url)

    # We should still have two pages
    assert len(pages) == 2
    assert base_url in pages
    assert f"{base_url}/account-settings" in pages
    assert f"{base_url}/password-reset" not in pages

    # Complete the indexing process
    parser = Parser()
    parsed_contents = [parser.parse_page(url, html) for url, html in pages.items()]

    indexer = Indexer(mock_config.indexing)
    chunks = indexer.process_documents(parsed_contents)

    assert len(chunks) > 0

    # We should still be able to query about account settings
    storage = FAISSStorage(dimension=mock_config.indexing.vector_dimension)
    storage.add_chunks(chunks)

    query_engine = QueryEngine(
        vector_store=storage,
        top_k=mock_config.query.top_k,
        confidence_threshold=mock_config.query.confidence_threshold
    )

    result = query_engine.process_query("Tell me about account settings")
    assert result.answer is not None
    assert "profile" in result.answer.lower() or "settings" in result.answer.lower()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])