"""
Pydantic validators for common patterns.
"""

import re
from pydantic import validator

def validate_company_name(name: str) -> str:
    if not re.match(r"^[a-zA-Z0-9\s\-\.,&]+$", name):
        raise ValueError("Invalid characters in company name")
    return name

def validate_symbol(symbol: str) -> str:
    if not re.match(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$", symbol):
        raise ValueError("Invalid market symbol format")
    return symbol
