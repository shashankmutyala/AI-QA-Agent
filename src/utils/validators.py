import re
import logging
from urllib.parse import urlparse
import requests
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """
    Normalize the URL by removing trailing slashes and cleaning whitespace.

    Args:
        url: URL to normalize

    Returns:
        Normalized URL string
    """
    return url.strip().rstrip('/')


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.

    Args:
        url: URL to validate

    Returns:
        Boolean indicating if the URL is valid
    """
    try:
        url = normalize_url(url)
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.error(f"[is_valid_url] Error validating URL {url}: {str(e)}")
        return False


def is_help_website(url: str) -> bool:
    """
    Check if a URL appears to be a help/documentation website.

    Args:
        url: URL to check

    Returns:
        Boolean indicating if the URL appears to be a help website
    """
    url = normalize_url(url)
    help_patterns = [
        r'(help|support|docs?|documentation|knowledge|faq|kb)[./]',
        r'/help|/docs|/support|/documentation|/faq|/kb',
    ]
    for pattern in help_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def check_url_accessibility(url: str, timeout: int = 5, retries: int = 3) -> Tuple[bool, Optional[str]]:
    """
    Check if a URL is accessible.

    Args:
        url: URL to check
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Tuple of (is_accessible, error_message)
    """
    url = normalize_url(url)
    for attempt in range(retries):
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            if response.status_code < 400:
                return True, None
            return False, f"Server returned error status: {response.status_code}"
        except requests.exceptions.Timeout:
            error_msg = "The request timed out"
        except requests.exceptions.ConnectionError:
            error_msg = "Failed to establish a connection"
        except requests.exceptions.RequestException as e:
            error_msg = str(e)

        logger.warning(f"[check_url_accessibility] Retry {attempt + 1}/{retries} failed for URL {url}: {error_msg}")

        # Only continue if we have more retries
        if attempt == retries - 1:
            return False, f"Failed after {retries} retries: {error_msg}"

    # This should not be reached but added for completeness
    return False, "Failed after retries"


def validate_query(query: str, min_length: int = 3, max_length: int = 500,
                   require_question_format: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate a user query.

    Args:
        query: The query to validate
        min_length: Minimum query length
        max_length: Maximum query length
        require_question_format: Whether to enforce question format

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query.strip():
        return False, "Query cannot be empty"

    query_length = len(query.strip())

    if query_length < min_length:
        return False, f"Query is too short (minimum {min_length} characters)"

    if query_length > max_length:
        return False, f"Query is too long (maximum {max_length} characters)"

    # Only check question format if explicitly required
    if require_question_format:
        if not re.match(r'(how|what|where|why|when|who|does|is|can|should|will|would|could).*[\?]?$', query.strip(),
                        re.IGNORECASE):
            return False, "Please phrase your query as a question (starting with words like 'how', 'what', 'where', etc.)"

    return True, None


def get_site_domain(url: str) -> Optional[str]:
    """
    Extract the domain from a URL.

    Args:
        url: URL to extract domain from

    Returns:
        Domain string or None if invalid URL
    """
    try:
        url = normalize_url(url)
        parsed_url = urlparse(url)
        return parsed_url.netloc
    except Exception as e:
        logger.error(f"Error extracting domain from URL {url}: {str(e)}")
        return None