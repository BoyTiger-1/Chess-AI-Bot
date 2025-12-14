"""AI Business Assistant data platform.

This package provides connectors, validations, transformations, feature engineering,
and a small DuckDB-based warehouse layer. Orchestration is implemented with Prefect
(when installed) but modules are designed to be usable without Prefect.
"""

from .config import Settings

__all__ = ["Settings"]
