from typing import TypedDict, List, Dict
from langchain.schema import Document


class TechState(TypedDict):
    """에이전트의 상태를 정의하는 클래스"""
    startup_names: List[str]
    current_startup: str
    web_data: str
    retrieved_docs: List[Document]
    tech_evaluations: List[Dict]
    processing_index: int
    vectorstore_ready: bool


