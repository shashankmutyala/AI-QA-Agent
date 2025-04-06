"""
Utilities Module
This module provides helper functions for validation, error handling, and other reusable utilities.
Exports:
    - is_valid_url: Validate the structure of URLs
    - is_help_website: Check if a URL belongs to a help/documentation site
    - check_url_accessibility: Test if a URL is accessible
    - validate_query: Validate user queries for proper formatting
    - handle_request_error: Handle HTTP request-related errors
    - handle_parsing_error: Handle errors during parsing
    - handle_storage_error: Handle errors related to data storage
    - safe_execute: Safely execute functions with error catching
    - log_exceptions: Decorator for logging exceptions in functions
    - format_error_for_display: Prepare errors for user display
"""

# Import validators
from .validators import (
    is_valid_url,
    is_help_website,
    check_url_accessibility,
    validate_query
)

# Import error handling utilities
from .error_handling import (
    handle_request_error,
    handle_parsing_error,
    handle_storage_error,
    safe_execute,
    log_exceptions,
    format_error_for_display
)

__all__ = [
    "is_valid_url",
    "is_help_website",
    "check_url_accessibility",
    "validate_query",
    "handle_request_error",
    "handle_parsing_error",
    "handle_storage_error",
    "safe_execute",
    "log_exceptions",
    "format_error_for_display"
]