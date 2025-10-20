from __future__ import annotations

from typing import TypedDict


class ComparisonState(TypedDict, total=False):
    """Typed state for company comparison."""

    company_name: str
    competitor_score: int
    competitor_analysis_basis: str


AgentState = ComparisonState


__all__ = ["ComparisonState", "AgentState"]
