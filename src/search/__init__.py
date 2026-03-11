# src/search/__init__.py
"""
Search package exports for SmartCampus V2T.

Purpose:
- Expose public index build and load entrypoints.
- Expose the query engine used by backend and evaluation flows.
"""

from .builder import build_or_update_index, load_index
from .engine import QueryEngine

__all__ = ["build_or_update_index", "load_index", "QueryEngine"]
