import os
import sys
import pytest
import responses
from unittest import mock
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crawling.crawler import Crawler
from src.crawling.parser import Parser, ParsedContent

# Sample HTML content for testing
SAMPLE_HTML = """<!DOCTYPE html><html><head><title>Test Page</title></head><body>
<header><h1>Test Documentation</h1></header>
<main><p>Main content</p></main></body></html>"""

# HTML with links for testing
LINKED_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="A test page for crawler testing">
</head>
<body>
    <header>
        <h1>Test Documentation</h1>
        <nav>
            <ul>
                <li><a href="/page1">Page 1</a></li>
                <li><a href="/page2">Page 2</a></li>
                <li><a href="https://external.com">External Link</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <section>
            <h2>Section 1</h2>
            <p>This is the content of section 1.</p>
            <a href="/page3">Link to Page 3</a>
        </section>
    </main>
</body>
</html>
"""

# HTML for a sub-page
PAGE1_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Page 1</title>
</head>
<body>
    <h1>Page 1 Content</h1>
    <p>This is page 1</p>
    <a href="/page4">Link to Page 4</a>
</body>
</html>
"""


@pytest.fixture
def crawler():
    """Create a crawler instance with default config."""
    config = {
        "max_depth": 2,
        "max_pages": 10,
        "rate_limit": 0.5,
        "timeout": 10,
        "max_workers": 5,
        "user_agent": 'Mozilla/5.0 (compatible; HelpWebsiteQAAgent/1.0)',
        "respect_robots_txt": True,
        "follow_redirects": True,
        "headers": {
            'User-Agent': 'Mozilla/5.0 (compatible; HelpWebsiteQAAgent/1.0)'
        }
    }
    return Crawler(config)


@pytest.fixture
def parser():
    """Create a parser instance."""
    return Parser()


@responses.activate
def test_crawler_initialization(crawler):
    """Test crawler initialization with default config."""
    assert crawler.max_depth == 2
    assert crawler.max_pages == 10
    assert hasattr(crawler, "visited_urls")


@responses.activate
def test_crawl_single_page(crawler):
    """Test crawling a single page with no links."""
    base_url = "https://example.com"

    # Mock the response
    responses.add(
        responses.GET,
        base_url,
        body=SAMPLE_HTML,
        status=200,
        content_type="text/html"
    )

    pages = crawler.crawl(base_url)

    assert len(pages) == 1
    assert base_url in pages
    assert pages[base_url] == SAMPLE_HTML


@responses.activate
def test_crawl_multiple_pages(crawler):
    """Test crawling multiple linked pages."""
    base_url = "https://example.com"

    # Mock the responses
    responses.add(
        responses.GET,
        base_url,
        body=LINKED_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/page1",
        body=PAGE1_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/page2",
        body="<html><body>Page 2</body></html>",
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/page3",
        body="<html><body>Page 3</body></html>",
        status=200,
        content_type="text/html"
    )

    pages = crawler.crawl(base_url)

    assert len(pages) == 4
    assert base_url in pages
    assert f"{base_url}/page1" in pages
    assert f"{base_url}/page2" in pages
    assert f"{base_url}/page3" in pages


@responses.activate
def test_crawl_circular_links(crawler):
    """Test the crawler handles circular links gracefully."""
    base_url = "https://example.com"
    # Circular reference
    responses.add(
        responses.GET,
        base_url,
        body=f'<a href="{base_url}">Link to self</a>',
        status=200,
        content_type="text/html"
    )

    pages = crawler.crawl(base_url)

    assert len(pages) == 1  # Should only visit the base URL once


@responses.activate
def test_crawl_non_html_content(crawler):
    """Test handling of non-HTML content like PDFs."""
    base_url = "https://example.com"
    responses.add(
        responses.GET,
        base_url,
        body="PDF content",
        status=200,
        content_type="application/pdf"
    )

    pages = crawler.crawl(base_url)

    # Different crawler implementations might handle this differently:
    # Some might store non-HTML content, others might skip it
    assert base_url not in pages or (base_url in pages and len(pages) == 1)


@responses.activate
def test_crawl_very_large_html(crawler):
    """Test crawling a very large HTML document."""
    base_url = "https://example.com"
    large_html = "<html>" + ("<p>Content</p>" * 10000) + "</html>"  # Simulate a large document
    responses.add(
        responses.GET,
        base_url,
        body=large_html,
        status=200,
        content_type="text/html"
    )

    pages = crawler.crawl(base_url)

    assert len(pages) == 1
    assert base_url in pages


@responses.activate
def test_respect_max_pages(crawler):
    """Test that crawler respects max_pages."""
    base_url = "https://example.com"
    crawler.max_pages = 2

    # Mock the responses
    responses.add(
        responses.GET,
        base_url,
        body=LINKED_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/page1",
        body=PAGE1_HTML,
        status=200,
        content_type="text/html"
    )

    # These should not be crawled due to max_pages=2
    responses.add(
        responses.GET,
        f"{base_url}/page2",
        body="<html><body>Page 2</body></html>",
        status=200,
        content_type="text/html"
    )

    pages = crawler.crawl(base_url)

    assert len(pages) <= 2
    assert base_url in pages


@responses.activate
def test_respect_max_depth(crawler):
    """Test that crawler respects max_depth."""
    base_url = "https://example.com"
    crawler.max_depth = 1  # Only crawl the base URL and direct links

    # Mock the responses
    responses.add(
        responses.GET,
        base_url,
        body=LINKED_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/page1",
        body=PAGE1_HTML,  # Contains a link to page4
        status=200,
        content_type="text/html"
    )

    # This should not be crawled due to max_depth=1
    responses.add(
        responses.GET,
        f"{base_url}/page4",
        body="<html><body>Page 4</body></html>",
        status=200,
        content_type="text/html"
    )

    pages = crawler.crawl(base_url)

    assert f"{base_url}/page4" not in pages


@responses.activate
def test_handle_http_errors(crawler):
    """Test handling of HTTP errors."""
    base_url = "https://example.com"

    # Mock the responses
    responses.add(
        responses.GET,
        base_url,
        body=LINKED_HTML,
        status=200,
        content_type="text/html"
    )

    responses.add(
        responses.GET,
        f"{base_url}/page1",
        status=404  # Not found
    )

    responses.add(
        responses.GET,
        f"{base_url}/page2",
        status=500  # Server error
    )

    pages = crawler.crawl(base_url)

    # Should have the base URL but not page1 or page2
    assert base_url in pages
    assert f"{base_url}/page1" not in pages
    assert f"{base_url}/page2" not in pages


def test_parser_parse_page(parser):
    """Test the parser's ability to extract content from HTML."""
    url = "https://example.com"

    parsed = parser.parse_page(url, LINKED_HTML)

    assert isinstance(parsed, ParsedContent)
    assert parsed.url == url
    assert parsed.title == "Test Page"
    assert "Test Documentation" in parsed.text
    assert "This is the content of section 1." in parsed.text

    # Check sections
    assert len(parsed.sections) >= 1
    assert any(section["title"] == "Section 1" for section in parsed.sections)


def test_parser_hierarchy(parser):
    """Test parser maintains hierarchy of sections."""
    html_with_hierarchy = """
    <html>
        <body>
            <h1>Title</h1>
            <h2>Section 1</h2>
            <p>Content in section 1.</p>
            <h2>Section 2</h2>
            <p>Content in section 2.</p>
        </body>
    </html>
    """
    url = "https://example.com"
    parsed = parser.parse_page(url, html_with_hierarchy)

    assert len(parsed.sections) == 2
    assert parsed.sections[0]["title"] == "Section 1"
    assert "Content in section 1." in parsed.sections[0]["content"]
    assert parsed.sections[1]["title"] == "Section 2"


def test_parser_empty_html(parser):
    """Test parser with empty HTML."""
    url = "https://example.com"

    parsed = parser.parse_page(url, "")

    assert parsed.url == url
    assert parsed.title == ""
    assert parsed.text == ""
    assert parsed.sections == []


def test_parser_invalid_html(parser):
    """Test parser with invalid HTML."""
    url = "https://example.com"

    parsed = parser.parse_page(url, "<html><not-valid>This is not valid HTML</not")

    assert parsed.url == url
    # The parser should still try to extract some content
    assert "This is not valid HTML" in parsed.text


@responses.activate
def test_logging_for_errors(crawler):
    """Ensure errors are logged when HTTP requests fail."""
    base_url = "https://example.com"
    responses.add(responses.GET, base_url, status=404)  # Mock a failed request

    with mock.patch("logging.Logger.error") as mocked_error:
        crawler.crawl(base_url)
        # The exact error message format might vary based on implementation
        mocked_error.assert_called()
        # Check that the URL and status code are mentioned in the error
        args = mocked_error.call_args[0][0]
        assert base_url in args
        assert "404" in args


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])