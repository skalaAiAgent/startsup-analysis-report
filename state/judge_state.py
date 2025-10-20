from typing import TypedDict


class JudgeState(TypedDict):
    decision: bool
    reason: str
    score: float