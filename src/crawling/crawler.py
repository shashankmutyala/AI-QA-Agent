import time
import logging
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from typing import Dict, Set, List, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Crawler:
    """Web crawler for help websites."""

    def __init__(self, config):
        """Initialize crawler with configuration."""
        # Extract configuration values
        self.max_depth = getattr(config, 'max_depth', 3)
        self.max_pages = getattr(config, 'max_pages', 100)
        self.rate_limit = getattr(config, 'rate_limit', 0.5)
        self.timeout = getattr(config, 'timeout', 10)
        self.max_workers = getattr(config, 'max_workers', 4)
        self.follow_redirects = getattr(config, 'follow_redirects', True)
        self.respect_robots_txt = getattr(config, 'respect_robots_txt', True)

        # Set default headers if not provided in config
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; HelpWebsiteQAAgent/1.0)'
        }
        self.headers = getattr(config, 'headers', default_headers)

        # Initialize tracking state
        self.visited_urls = set()
        self.base_domain = None  # Will be set during crawl

    def crawl(self, base_url: str) -> Dict[str, str]:
        """
        Crawl a website starting from base_url.

        Args:
            base_url: Starting URL for crawling

        Returns:
            Dictionary mapping URLs to HTML content
        """
        self.base_domain = urlparse(base_url).netloc
        pages = {}

        # Check URL validity
        if not self._is_valid_url(base_url):
            logger.error(f"Invalid base URL: {base_url}")
            return pages

        # Start recursive crawl
        self._crawl_recursive(base_url, pages, depth=0)

        logger.info(f"Crawling complete. Visited {len(self.visited_urls)} URLs. Extracted {len(pages)} pages.")
        return pages

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except:
            return False

    def _should_crawl(self, url: str) -> bool:
        """Determine if a URL should be crawled."""
        parsed = urlparse(url)

        # Check if URL is already visited
        if url in self.visited_urls:
            return False

        # Check if URL is in the same domain
        if parsed.netloc and parsed.netloc != self.base_domain:
            return False

        # Skip non-HTML content
        if any(url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js']):
            return False

        return True

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch content from URL."""
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=self.follow_redirects
            )

            if response.status_code != 200:
                logger.error(f"Error fetching {url}: HTTP {response.status_code}")
                return None

            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower():
                logger.info(f"Skipping non-HTML content: {url}")
                return None

            return response.text

        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def _extract_links(self, url: str, html_content: str) -> List[str]:
        """Extract links from HTML content."""
        links = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for anchor in soup.find_all('a', href=True):
                href = anchor['href']
                if not href or href.startswith('#'):
                    continue

                absolute_url = urljoin(url, href)
                if self._should_crawl(absolute_url):
                    links.append(absolute_url)
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {str(e)}")

        return links

    def _crawl_recursive(self, url: str, pages: Dict[str, str], depth: int) -> None:
        """
        Recursively crawl pages starting from url.

        Args:
            url: Current URL to crawl
            pages: Dictionary to store results
            depth: Current crawl depth
        """
        # Stop conditions
        if depth > self.max_depth or len(pages) >= self.max_pages or url in self.visited_urls:
            return

        # Mark as visited
        self.visited_urls.add(url)

        # Fetch page
        html_content = self._fetch_page(url)
        if not html_content:
            return

        # Store page
        pages[url] = html_content

        # Apply rate limiting
        if self.rate_limit > 0:
            time.sleep(self.rate_limit)

        # Extract and follow links
        links = self._extract_links(url, html_content)
        for link in links:
            if len(pages) >= self.max_pages:
                break
            self._crawl_recursive(link, pages, depth + 1)