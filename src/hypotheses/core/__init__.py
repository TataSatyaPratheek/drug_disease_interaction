"""
core package – common utilities for all hypothesis modules.
Exports the abstract `Hypothesis` base-class.
"""
from .base import Hypothesis  # re-export for convenience

__all__ = ["Hypothesis"]
