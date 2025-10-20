import os
from langchain_openai import ChatOpenAI
from state.final_state import FinalState
from state.judge_state import JudgeState


class JudgeAgent:
    """보고서 생성 여부를 판단하는 Agent"""
    
    def __init__(self, final_state: FinalState):
        self.final_state = final_state
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.structured_llm = self.llm.with_structured_output(JudgeState)
        self.threshold = 75.0
        self.judge_result: JudgeState | None = None
    
    def should_generate_report(self) -> bool:
        """보고서 생성 여부 판단"""
        if self.judge_result is None:
            self.judge_result = self._evaluate()
        
        return self.judge_result["decision"]
    
    def get_rejection_reason(self) -> str:
        """Reject 이유 반환"""
        if self.judge_result is None:
            self.judge_result = self._evaluate()
        
        return self.judge_result["reason"]
    
    def _evaluate(self) -> JudgeState:
        """평가 실행"""
        tech_state = self.final_state["tech"]
        tech_evaluations = tech_state["tech_evaluations"]
        
        # tech_score 추출
        tech_score = tech_evaluations["tech_score"]
        
        # 1단계: 점수 체크
        if tech_score < self.threshold:
            return {
                "decision": False,
                "reason": f"기술 점수가 기준 미달입니다 ({tech_score}점 < {self.threshold}점)",
                "score": tech_score
            }
        
        # 2단계: GPT-4o 검증
        try:
            gpt_result = self._validate_with_gpt(tech_state, tech_score)
            return gpt_result
            
        except Exception as e:
            return {
                "decision": False,
                "reason": f"GPT-4o 검증 중 오류 발생: {str(e)}",
                "score": tech_score
            }
    
    def _validate_with_gpt(self, tech_state: dict, tech_score: float) -> JudgeState:
        """GPT-4o를 통한 평가 검증"""
        
        prompt = f"""다음은 "{self.final_state['company_name']}"에 대한 기술 평가 결과입니다:

=== 기술 평가 결과 ===
회사명: {tech_state.get('current_startup', 'N/A')}
기술 점수: {tech_score}점
평가 근거: {tech_state['tech_evaluations'].get('score_basis', 'N/A')}

=== 추가 정보 ===
평가 대상 스타트업들: {tech_state.get('startup_names', [])}
웹 데이터 수집 여부: {'완료' if tech_state.get('web_data') else '미완료'}
문서 검색 수: {len(tech_state.get('retrieved_docs', []))}개

=== 1단계 평가 결과 ===
- 기술 점수: {tech_score}점
- 기준 점수: {self.threshold}점 이상
- 1단계 결과: 통과

=== 2단계 검증 요청 ===
위 평가 결과가 타당한지 종합적으로 판단해주세요:
1. 기술 점수가 적절한가?
2. 평가 근거가 합리적이고 구체적인가?
3. 전반적으로 신뢰할 수 있는 분석인가?

타당하면 decision을 true로, 그렇지 않으면 false로 반환하세요.
reason에는 구체적인 판단 이유를 한 문장으로 작성하세요.
score에는 현재 기술 점수 {tech_score}를 그대로 반환하세요.
"""
        
        result: JudgeState = self.structured_llm.invoke(prompt)
        
        return result