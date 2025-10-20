from typing import TypedDict

from state import comparison_state, market_state
from state.tech_state import TechState


class FinalState(TypedDict):
    """최종 통합 분석 결과"""
    company_name: str
    market: market_state.MarketState
    tech: TechState
    comparison: comparison_state.ComparisonState