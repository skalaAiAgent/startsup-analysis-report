from typing import TypedDict


class FinalState(TypedDict):
    """최종 통합 분석 결과"""
    company_name: str
    market: MarketState
    tech: TechState
    comparison: ComparisonState