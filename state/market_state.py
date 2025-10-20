from __future__ import annotations

from typing import TypedDict


class MarketState(TypedDict, total=False):
    """Typed state for market evaluation Agent.

    NOTE: 필드명은 비교 에이전트의 상태 스키마와 호환되도록
    요청에 맞춰 다음과 같이 지정합니다.
      - company_name                (원래 startup_name)
      - competitor_score            (원래 market_score)
      - competitor_analysis_basis   (원래 rationale)
    """

    company_name: str
    competitor_score: int
    competitor_analysis_basis: str


AgentState = MarketState

__all__ = ["MarketState", "AgentState"]
