"""Segment-level active learning query strategies."""

from __future__ import annotations

from .base import QueryStrategy
from .registry import STRATEGY_NAMES, build_query_strategy

__all__ = ["QueryStrategy", "STRATEGY_NAMES", "build_query_strategy"]
