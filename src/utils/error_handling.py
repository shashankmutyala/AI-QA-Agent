import logging
import sys
import traceback
import time
import requests
from typing import Dict, Any, Optional, TypeVar, Callable, Union, Type
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


def handle_request_error(error: Exception) -> Dict[str, Any]:
    """
    Handle errors related to HTTP requests and return a standardized error response.

    Args:
        error: The exception to handle

    Returns:
        Dictionary with error code and user-friendly message
    """
    error_map = {
        requests.exceptions.Timeout: (
        "E001", "The request timed out. The server might be experiencing high load or the site might be down."),
        requests.exceptions.ConnectionError: (
        "E002", "Failed to establish a connection. Please check your internet connection or the website URL."),
        requests.exceptions.HTTPError: ("E003", "HTTP error occurred. The server might be experiencing issues."),
        requests.exceptions.TooManyRedirects: (
        "E004", "Too many redirects. The website structure might be problematic."),
        requests.exceptions.RequestException: ("E005", "A request error occurred."),
    }

    # Check for more specific error details
    if isinstance(error, requests.exceptions.HTTPError) and hasattr(error, 'response'):
        status_code = error.response.status_code
        return {
            "error_code": f"E003-{status_code}",
            "message": f"HTTP error {status_code} occurred. The server might be experiencing issues.",
            "status_code": status_code
        }

    code, message = error_map.get(type(error), ("E999", f"An unexpected error occurred: {str(error)}"))
    return {"error_code": code, "message": message}


def handle_parsing_error(error: Exception, url: Optional[str] = None) -> Dict[str, Any]:
    """
    Handle errors related to content parsing and return a standardized error response.

    Args:
        error: The exception to handle
        url: The URL being parsed (optional)

    Returns:
        Dictionary with error code and user-friendly message
    """
    context = f" when parsing {url}" if url else ""
    error_map = {
        ValueError: ("E101", f"Invalid content format{context}: {str(error)}"),
        AttributeError: (
        "E102", f"Failed to parse content structure{context}. The website might have an unexpected layout."),
        KeyError: ("E103", f"Missing expected data{context}: {str(error)}"),
        IndexError: ("E104", f"Data structure mismatch{context}: {str(error)}"),
        TypeError: ("E105", f"Unexpected data type{context}: {str(error)}"),
    }
    code, message = error_map.get(type(error), ("E199", f"An error occurred{context}: {str(error)}"))
    return {"error_code": code, "message": message}


def handle_storage_error(error: Exception) -> Dict[str, Any]:
    """
    Handle errors related to data storage and retrieval.

    Args:
        error: The exception to handle

    Returns:
        Dictionary with error code and user-friendly message
    """
    error_map = {
        FileNotFoundError: ("E201", "The specified file or directory was not found."),
        PermissionError: ("E202", "Permission denied. Please check file permissions."),
        IOError: ("E203", f"I/O error: {str(error)}"),
        OSError: ("E204", f"Operating system error: {str(error)}"),
        IsADirectoryError: ("E205", "Expected a file but found a directory."),
        NotADirectoryError: ("E206", "Expected a directory but found a file."),
    }
    code, message = error_map.get(type(error), ("E299", f"Storage error: {str(error)}"))
    return {"error_code": code, "message": message}


def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is potentially recoverable with a retry.

    Args:
        error: The exception to check

    Returns:
        Boolean indicating if the error might be resolved by retrying
    """
    recoverable_errors = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
        IOError,
        TimeoutError,
    )

    # Consider certain HTTP status codes as recoverable
    if isinstance(error, requests.exceptions.HTTPError) and hasattr(error, 'response'):
        # 429 (Too Many Requests), 500, 502, 503, 504 (Server errors) are recoverable
        recoverable_status_codes = {429, 500, 502, 503, 504}
        return error.response.status_code in recoverable_status_codes

    return isinstance(error, recoverable_errors)


def safe_execute(func: Callable[..., T], *args, retries: int = 3,
                 retry_delay: float = 1.0, backoff_factor: float = 2.0,
                 recoverable_only: bool = True, **kwargs) -> Optional[T]:
    """
    Safely execute a function, catching and logging any exceptions.

    Args:
        func: Function to execute
        *args: Positional arguments
        retries: Number of retries for recoverable errors
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for the delay between consecutive retries
        recoverable_only: Only retry on recoverable errors
        **kwargs: Keyword arguments

    Returns:
        Function result or None if an exception occurred
    """
    delay = retry_delay

    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Determine if we should retry
            should_retry = (attempt < retries - 1) and (not recoverable_only or is_recoverable_error(e))

            if should_retry:
                logger.warning(f"[safe_execute] Attempt {attempt + 1}/{retries} failed for {func.__name__}: {str(e)}")
                logger.info(f"[safe_execute] Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                delay *= backoff_factor  # Exponential backoff
            else:
                logger.error(f"[safe_execute] {'All retries failed' if attempt > 0 else 'Execution failed'} "
                             f"for {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
                return None

    return None


def log_exceptions(func: Callable) -> Callable:
    """
    Decorator to log exceptions from a function without stopping execution.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    return wrapper


def format_error_for_display(error: Union[str, Exception, Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]:
    """
    Format an error message or exception for display to the user.

    Args:
        error: The error message, exception, or error dictionary
        verbose: Whether to include detailed information

    Returns:
        Dictionary with formatted error information
    """
    # Handle different types of error inputs
    if isinstance(error, dict) and "message" in error:
        result = {
            "error": True,
            "message": error["message"],
            "error_code": error.get("error_code", "E000")
        }
    elif isinstance(error, Exception):
        result = {
            "error": True,
            "message": str(error),
            "error_code": "E000"
        }
    else:
        result = {
            "error": True,
            "message": str(error),
            "error_code": "E000"
        }

    if verbose:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type:
            result["exception_type"] = exc_type.__name__
            result["details"] = str(exc_value)
            result["traceback"] = traceback.format_exc()

    return result