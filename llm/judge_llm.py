import os
from langchain_openai import ChatOpenAI
from state.final_state import FinalState
from state.judge_state import JudgeState


class JudgeLLM:
    """보고서 생성 여부를 판단하는 Agent"""
    
    def __init__(self, final_state: FinalState):
        self.final_state = final_state
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.structured_llm = self.llm.with_structured_output(JudgeState)
        self.threshold = 75.0  # 평균 75점 기준
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
        market_state = self.final_state.get("market")
        tech_state = self.final_state.get("tech")
        comparison_state = self.final_state.get("comparison")
        
        # 점수 추출
        scores = []
        score_details = []
        
        if market_state:
            market_score = market_state.get("competitor_score", 0)
            scores.append(market_score)
            score_details.append(f"Market: {market_score}점")
        
        if tech_state:
            tech_score = tech_state.get("technology_score", 0)
            scores.append(tech_score)
            score_details.append(f"Tech: {tech_score}점")
        
        if comparison_state:
            comparison_score = comparison_state.get("competitor_score", 0)
            scores.append(comparison_score)
            score_details.append(f"Comparison: {comparison_score}점")
        
        # 평가 가능한 데이터가 없는 경우
        if not scores:
            return {
                "decision": False,
                "reason": "평가할 데이터가 없습니다 (market, tech, comparison 모두 None)",
                "score": 0.0
            }
        
        # 평균 점수 계산
        average_score = sum(scores) / len(scores)
        
        # 1단계: 평균 점수 체크
        if average_score < self.threshold:
            score_detail_str = ", ".join(score_details)
            return {
                "decision": False,
                "reason": f"평균 점수가 기준 미달입니다 (평균 {average_score:.1f}점 < {self.threshold}점) [{score_detail_str}]",
                "score": average_score
            }
        
        # 2단계: GPT-4o 검증
        try:
            gpt_result = self._validate_with_gpt(
                market_state, 
                tech_state, 
                comparison_state, 
                average_score,
                score_details
            )
            return gpt_result
            
        except Exception as e:
            return {
                "decision": False,
                "reason": f"GPT-4o 검증 중 오류 발생: {str(e)}",
                "score": average_score
            }
    
    def _validate_with_gpt(
        self, 
        market_state: dict | None, 
        tech_state: dict | None, 
        comparison_state: dict | None,
        average_score: float,
        score_details: list[str]
    ) -> JudgeState:
        """GPT-4o를 통한 평가 검증"""
        
        # 평가 결과 섹션 구성
        evaluation_sections = []
        
        if market_state:
            market_section = f"""=== Market 평가 결과 ===
회사명: {market_state.get('company_name', 'N/A')}
경쟁사 점수: {market_state.get('competitor_score', 0)}점
평가 근거: {market_state.get('competitor_analysis_basis', 'N/A')}
"""
            evaluation_sections.append(market_section)
        
        if tech_state:
            tech_section = f"""=== Tech 평가 결과 ===
회사명: {tech_state.get('company_name', 'N/A')}
기술 점수: {tech_state.get('technology_score', 0)}점
평가 근거: {tech_state.get('technology_analysis_basis', 'N/A')}

항목별 점수:
- 혁신성(Innovation): {tech_state.get('category_scores', {}).get('innovation', 0)}점 / 30점
- 완성도(Completeness): {tech_state.get('category_scores', {}).get('completeness', 0)}점 / 30점
- 경쟁력(Competitiveness): {tech_state.get('category_scores', {}).get('competitiveness', 0)}점 / 20점
- 특허(Patent): {tech_state.get('category_scores', {}).get('patent', 0)}점 / 10점
- 확장성(Scalability): {tech_state.get('category_scores', {}).get('scalability', 0)}점 / 10점
"""
            evaluation_sections.append(tech_section)
        
        if comparison_state:
            comparison_section = f"""=== Comparison 평가 결과 ===
회사명: {comparison_state.get('company_name', 'N/A')}
경쟁사 비교 점수: {comparison_state.get('competitor_score', 0)}점
평가 근거: {comparison_state.get('competitor_analysis_basis', 'N/A')}
"""
            evaluation_sections.append(comparison_section)
        
        prompt = f"""다음은 "{self.final_state['company_name']}"에 대한 종합 평가 결과입니다:

{chr(10).join(evaluation_sections)}

=== 1단계 평가 결과 ===
- 개별 점수: {", ".join(score_details)}
- 평균 점수: {average_score:.1f}점
- 기준 점수: {self.threshold}점 이상
- 1단계 결과: 통과

=== 2단계 검증 요청 ===
위 평가 결과가 타당한지 종합적으로 판단해주세요:
1. 각 영역의 점수가 적절한가?
2. 평가 근거들이 합리적이고 구체적인가?
3. 전반적으로 신뢰할 수 있는 분석인가?
4. 평균 점수가 실제 회사의 역량을 잘 반영하는가?

타당하면 decision을 true로, 그렇지 않으면 false로 반환하세요.
reason에는 구체적인 판단 이유를 한 문장으로 작성하세요.
score에는 현재 평균 점수 {average_score:.1f}를 그대로 반환하세요.
"""
        
        result: JudgeState = self.structured_llm.invoke(prompt)
        
        return result