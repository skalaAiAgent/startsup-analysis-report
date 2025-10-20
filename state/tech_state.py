from typing import TypedDict, List, Dict
# ========== TypedDict 정의 ==========

class CategoryScoresDict(TypedDict):
    """항목별 점수"""
    innovation: int  # 0-30
    completeness: int  # 0-30
    competitiveness: int  # 0-20
    patent: int  # 0-10
    scalability: int  # 0-10


class TechState(TypedDict):
    """기술 평가 결과"""
    company_name: str
    category_scores: CategoryScoresDict
    technology_score: int  # 0-100
    technology_analysis_basis: str
