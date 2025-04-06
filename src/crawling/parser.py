import re
import logging
from bs4 import BeautifulSoup, Tag, NavigableString
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import html2text


@dataclass
class ParsedContent:
    """Data class for storing parsed content from a page."""
    url: str
    title: str
    text: str
    sections: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return f"ParsedContent(url={self.url}, title={self.title}, text_length={len(self.text)}, sections={len(self.sections)})"


class Parser:
    """
    HTML parser that extracts meaningful content from help website pages.
    """

    def __init__(self):
        """Initialize the parser."""
        self.logger = logging.getLogger(__name__)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.ignore_tables = False
        self.h2t.body_width = 0  # No wrapping

        # Common selectors for main content areas
        self.content_selectors = [
            'article', 'main', '.content', '.main-content', '.documentation',
            '.article', '.post', '.entry-content', '.doc-content', '#content',
            '.help-content', '.knowledge-base', '.kb-article', '.support-article'
        ]

        # Elements likely to be boilerplate/navigation
        self.noise_selectors = [
            'nav', 'header', 'footer', '#header', '#footer', '.nav', '.navigation',
            '.menu', '.sidebar', '.footer', '.header', '.comment', '.advertisement',
            '.breadcrumb', '.breadcrumbs', '.search', '.pagination', '.related',
            '#nav', '#menu', '#sidebar', '.cookie-banner', '.banner', '.ad'
        ]

    def parse_page(self, url: str, html_content: str) -> ParsedContent:
        """
        Parse an HTML page and extract meaningful content.

        Args:
            url: URL of the page
            html_content: HTML content as a string

        Returns:
            ParsedContent object with extracted information
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for element in soup.find_all(['script', 'style']):
            element.decompose()

        # Remove noise elements (navigation, ads, etc.)
        for selector in self.noise_selectors:
            for element in soup.select(selector):
                element.decompose()

        title = self._extract_title(soup)
        main_content = self._extract_main_content(soup)
        sections = self._extract_sections(main_content)
        metadata = self._extract_metadata(soup)

        # Convert to plain text
        text = self.h2t.handle(str(main_content))

        return ParsedContent(
            url=url,
            title=title,
            text=text,
            sections=sections,
            metadata=metadata
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the title of the page."""
        article_title = soup.find(['h1', 'h2'], class_=['title', 'article-title', 'post-title', 'entry-title'])
        if article_title:
            return article_title.get_text(strip=True)

        if soup.title:
            title = soup.title.get_text(strip=True)
            return re.sub(r'\s*[|]\s*.+$|\s*[-]\s*.+$', '', title)

        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)

        return "Untitled Page"

    def _extract_main_content(self, soup: BeautifulSoup) -> Tag:
        """Extract the main content area of the page."""
        for selector in self.content_selectors:
            content = soup.select_one(selector)
            if content and self._is_substantial(content):
                return content

        article = soup.find(['article', 'main', 'section'])
        if article and self._is_substantial(article):
            return article

        divs = soup.find_all('div')
        if divs:
            largest_div = max(divs, key=lambda x: len(x.get_text(strip=True)))
            if self._is_substantial(largest_div):
                return largest_div

        body = soup.find('body')
        return body if body else soup

    def _is_substantial(self, element: Tag) -> bool:
        """Check if an element contains substantial content."""
        text = element.get_text(strip=True)
        if len(text) < 100:
            return False
        if not element.find_all(['p', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4']):
            return False
        return True

    def _extract_sections(self, content: Tag) -> List[Dict[str, Any]]:
        """Extract hierarchical sections from the content."""
        sections = []
        headings = content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if not headings:
            return [{
                'title': '',
                'level': 0,
                'content': self.h2t.handle(str(content)),
                'subsections': []
            }]

        for i, heading in enumerate(headings):
            heading_level = int(heading.name[1])
            heading_text = heading.get_text(strip=True)

            content_elements = []
            current = heading.next_sibling

            while current and (i == len(headings) - 1 or current != headings[i + 1]):
                if isinstance(current, Tag) and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    content_elements.append(str(current))
                current = current.next_sibling

            section_content = self.h2t.handle(''.join(content_elements))

            sections.append({
                'title': heading_text,
                'level': heading_level,
                'content': section_content,
                'subsections': []
            })

        self._organize_sections(sections)
        return sections

    def _organize_sections(self, sections: List[Dict[str, Any]]) -> None:
        """Organize sections into a hierarchical structure."""
        stack = []
        for section in sections:
            while stack and stack[-1]['level'] >= section['level']:
                stack.pop()
            if stack:
                parent = stack[-1]
                parent.setdefault('subsections', []).append(section)
            stack.append(section)

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from the page."""
        metadata = {}

        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name') and tag.get('content'):
                metadata[tag['name']] = tag['content']

        ld_json = soup.find('script', type='application/ld+json')
        if ld_json:
            metadata['structured_data'] = ld_json.string

        return metadata