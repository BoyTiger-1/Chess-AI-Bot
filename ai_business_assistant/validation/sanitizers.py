"""
Sanitization utilities for input data.
"""

import re
import html
from typing import Any, Dict, Optional

def sanitize_string(s: str) -> str:
    """Remove potentially dangerous characters from string."""
    if not s:
        return s
    # Strip null bytes
    s = s.replace('\x00', '')
    # Trim whitespace
    s = s.strip()
    return s

def escape_html(s: str) -> str:
    """Escape HTML characters."""
    return html.escape(s)

def strip_sql_injection(s: str) -> str:
    """Basic SQL injection prevention (though ORM is preferred)."""
    # Remove common SQL keywords used in injections if they appear in suspicious context
    # This is a very basic fallback
    suspicious_patterns = [
        r"--", r";", r"/\*", r"\*/", r"@@", r"char\(", r"nchar\(", r"varchar\("
    ]
    for pattern in suspicious_patterns:
        s = re.sub(pattern, "", s, flags=re.IGNORECASE)
    return s
