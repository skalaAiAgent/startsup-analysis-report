from typing import TypedDict
from state import FinalState


class WorkflowState(TypedDict):
      company_name: str
      final_state: FinalState
      judgment: bool
      report_path: str | None
      error: str | None