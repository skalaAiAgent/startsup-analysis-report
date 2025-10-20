from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from state.final_state import FinalState
from state.judge_state import JudgeState


load_dotenv()


class JudgeLLM:
    """보고서 생성 여부를 판단하는 Agent"""
    
    def __init__(self, final_state: FinalState):
        self.final_state = final_state
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.structured_llm = self.llm.with_structured_output(JudgeState)
        
        # Scorecard Method 기반 가중치
        self.weights = {
            "market": 0.40,      # 시장성 40%
            "tech": 0.25,        # 기술력 25%
            "comparison": 0.35   # 종합 경쟁력 35%
        }
        
        self.threshold = 80.0  # 엄격한 기준 (PSI VC Network 표준)
        self.judge_result: JudgeState | None = None
        
        # 가중치 설정 근거
        self.weight_rationale = """
가중치 설정 근거:

1. Market (40%): 시장성 최우선
   - Scorecard Method 표준 25% → 40%로 상향
   - 벤처투자에서 시장 규모가 가장 중요 (한재우 외, 2016: 33.9%)

2. Tech (25%): 기술 차별화
   - 표준 15% → 25%로 상향
   - 혁신성, 완성도, 경쟁력, 특허, 확장성 종합 평가

3. Comparison (35%): 실행력 검증
   - 트랙션(MAU) + 재무 + 투자금 복합 지표
   - 실제 시장 성과 증명

출처: Scorecard Valuation Method (Bill Payne, 2009) + 한재우 외(2016)
"""
    
    def should_generate_report(self) -> bool:
        """보고서 생성 여부 판단"""
        if self.judge_result is None:
            self.judge_result = self._evaluate()
        
        return self.judge_result.get("decision", False)
    
    def get_rejection_reason(self) -> str:
        """Reject 이유 반환"""
        if self.judge_result is None:
            self.judge_result = self._evaluate()
        
        return self.judge_result.get("reason", "이유 없음")
    
    def _evaluate(self) -> JudgeState:
        """가중 평균 기반 평가"""
        market_state = self.final_state.get("market")
        tech_state = self.final_state.get("tech")
        comparison_state = self.final_state.get("comparison")
        
        # 점수와 가중치 수집
        weighted_scores = []
        score_details = []
        
        if market_state:
            score = market_state.get("competitor_score", 0)
            weight = self.weights["market"]
            weighted_scores.append(score * weight)
            score_details.append(
                f"Market: {score}점 (가중치 {weight*100:.0f}%)"
            )
        
        if tech_state:
            score = tech_state.get("technology_score", 0)
            weight = self.weights["tech"]
            weighted_scores.append(score * weight)
            score_details.append(
                f"Tech: {score}점 (가중치 {weight*100:.0f}%)"
            )
        
        if comparison_state:
            score = comparison_state.get("competitor_score", 0)
            weight = self.weights["comparison"]
            weighted_scores.append(score * weight)
            score_details.append(
                f"Comparison: {score}점 (가중치 {weight*100:.0f}%)"
            )
        
        if not weighted_scores:
            return {
                "decision": False,
                "reason": "평가할 데이터가 없습니다",
                "score": 0.0
            }
        
        # 가중 평균 계산
        weighted_average = sum(weighted_scores)
        
        # 1단계: 가중 평균 점수 체크
        if weighted_average < self.threshold:
            detail_str = ", ".join(score_details)
            return {
                "decision": False,
                "reason": (
                    f"가중 평균 점수가 기준 미달입니다 "
                    f"(가중평균 {weighted_average:.1f}점 < {self.threshold}점)\n"
                    f"[{detail_str}]\n"
                    f"※ 가중치는 Scorecard Method 기반"
                ),
                "score": weighted_average
            }
        
        # 2단계: GPT-4o 검증
        try:
            gpt_result = self._validate_with_gpt(
                market_state,
                tech_state,
                comparison_state,
                weighted_average,
                score_details
            )
            return gpt_result
            
        except Exception as e:
            return {
                "decision": False,
                "reason": f"GPT-4o 검증 중 오류 발생: {str(e)}",
                "score": weighted_average
            }
    
    def _validate_with_gpt(
        self,
        market_state: dict | None,
        tech_state: dict | None,
        comparison_state: dict | None,
        weighted_average: float,
        score_details: list[str]
    ) -> JudgeState:
        """GPT-4o를 통한 평가 검증"""
        
        # 평가 결과 섹션 구성
        evaluation_sections = []
        
        if market_state:
            market_section = f"""=== Market 평가 결과 ===
회사명: {market_state.get('company_name', 'N/A')}
시장성 점수: {market_state.get('competitor_score', 0)}점
평가 근거: {market_state.get('competitor_analysis_basis', 'N/A')}
"""
            evaluation_sections.append(market_section)
        
        if tech_state:
            tech_section = f"""=== Tech 평가 결과 ===
회사명: {tech_state.get('company_name', 'N/A')}
기술 점수: {tech_state.get('technology_score', 0)}점
평가 근거: {tech_state.get('technology_analysis_basis', 'N/A')}

항목별 점수:
- 혁신성: {tech_state.get('category_scores', {}).get('innovation', 0)}점 / 30점
- 완성도: {tech_state.get('category_scores', {}).get('completeness', 0)}점 / 30점
- 경쟁력: {tech_state.get('category_scores', {}).get('competitiveness', 0)}점 / 20점
- 특허: {tech_state.get('category_scores', {}).get('patent', 0)}점 / 10점
- 확장성: {tech_state.get('category_scores', {}).get('scalability', 0)}점 / 10점
"""
            evaluation_sections.append(tech_section)
        
        if comparison_state:
            comparison_section = f"""=== Comparison 평가 결과 ===
회사명: {comparison_state.get('company_name', 'N/A')}
종합 경쟁력 점수: {comparison_state.get('competitor_score', 0)}점
평가 근거: {comparison_state.get('competitor_analysis_basis', 'N/A')}
"""
            evaluation_sections.append(comparison_section)
        
        prompt = f"""다음은 "{self.final_state['company_name']}"에 대한 종합 평가 결과입니다:

{chr(10).join(evaluation_sections)}

=== 가중치 기반 평가 결과 ===
{self.weight_rationale}

- 개별 점수: {", ".join(score_details)}
- **가중 평균 점수: {weighted_average:.1f}점**
- 기준 점수: {self.threshold}점 이상
- 1단계 결과: 통과

=== 2단계 검증 요청 ===
위 가중치 기반 평가 결과가 타당한지 종합적으로 판단해주세요:
1. 각 영역의 점수가 적절한가?
2. 평가 근거들이 합리적이고 구체적인가?
3. 가중 평균 {weighted_average:.1f}점이 실제 회사의 역량을 잘 반영하는가?
4. 전반적으로 신뢰할 수 있는 분석인가?

타당하면 decision을 true로, 그렇지 않으면 false로 반환하세요.
reason에는 구체적인 판단 이유를 한 문장으로 작성하세요.
score에는 현재 가중 평균 점수 {weighted_average:.1f}를 그대로 반환하세요.
"""
        
        result: JudgeState = self.structured_llm.invoke(prompt)
        return result