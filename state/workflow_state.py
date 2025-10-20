from typing import TypedDict
from state import final_state


class WorkflowState(TypedDict):
      company_name: str
      final_state: final_state
      judgment: bool
      report_path: str | None
      error: str | None