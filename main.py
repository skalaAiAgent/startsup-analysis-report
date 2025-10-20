import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# LangGraph import
from langgraph.graph import StateGraph, END

# Agent import
from agents.market_evaluation_agent import MarketAgent
from agents.tech_agent import TechAgent
from agents.company_comparison_agent import ComparisonAgent
from llm.judge_llm import JudgeLLM
from llm.report_llm import ReportLLM

# State import
from state.market_state import MarketState
from state.tech_state import TechState
from state.comparison_state import ComparisonState
from state.final_state import FinalState
from state.workflow_state import WorkflowState


# ===== 상수 정의 =====
COMPANIES = ["트립소다", "트립비토즈", "하이어플레이스", "리브애니웨어", "어딩"]
REPORTS_DIR = "reports"


# ===== 유틸리티 함수 =====
def ensure_reports_dir():
    """reports 디렉토리 생성"""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        print(f"[INFO] {REPORTS_DIR} 디렉토리 생성됨")


def get_timestamp():
    """타임스탬프 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_judgment_to_file(company_name: str, final_state: FinalState, judgment: bool, reason: str, score: float) -> str:
    """판단 결과를 파일로 저장 (Reject인 경우에만 사용)"""
    timestamp = get_timestamp()
    filename = f"{company_name}_rejected_{timestamp}.txt"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    # 점수 정보 추출
    score_details = []
    if final_state.get('market'):
        market_score = final_state['market'].get('competitor_score', 0)
        score_details.append(f"Market: {market_score}점")
    
    if final_state.get('tech'):
        tech_score = final_state['tech'].get('technology_score', 0)
        score_details.append(f"Tech: {tech_score}점")
        
        # 기술 항목별 점수
        category_scores = final_state['tech'].get('category_scores', {})
        score_details.append("  기술 세부 점수:")
        score_details.append(f"    - 혁신성: {category_scores.get('innovation', 0)}/30")
        score_details.append(f"    - 완성도: {category_scores.get('completeness', 0)}/30")
        score_details.append(f"    - 경쟁력: {category_scores.get('competitiveness', 0)}/20")
        score_details.append(f"    - 특허: {category_scores.get('patent', 0)}/10")
        score_details.append(f"    - 확장성: {category_scores.get('scalability', 0)}/10")
    
    if final_state.get('comparison'):
        comparison_score = final_state['comparison'].get('competitor_score', 0)
        score_details.append(f"Comparison: {comparison_score}점")
    
    content = f"""회사명: {company_name}
분석 일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
판단 결과: REJECT ✗
평균 점수: {score:.1f}점

{"="*60}
점수 상세
{"="*60}
{chr(10).join(score_details)}

{"="*60}
판단 이유
{"="*60}
{reason}

{"="*60}
평가 근거
{"="*60}
"""
    
    # Market 평가 근거
    if final_state.get('market'):
        content += f"\n[Market 평가]\n{final_state['market'].get('competitor_analysis_basis', 'N/A')}\n"
    
    # Tech 평가 근거
    if final_state.get('tech'):
        content += f"\n[Tech 평가]\n{final_state['tech'].get('technology_analysis_basis', 'N/A')}\n"
    
    # Comparison 평가 근거
    if final_state.get('comparison'):
        content += f"\n[Comparison 평가]\n{final_state['comparison'].get('competitor_analysis_basis', 'N/A')}\n"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def save_report_to_file(company_name: str, report_content: str) -> str:
    """보고서를 파일로 저장 (Accept인 경우)"""
    timestamp = get_timestamp()
    filename = f"{company_name}_report_{timestamp}.md"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return filepath


# ===== 병렬 분석 실행 =====
async def run_market_agent(company_name: str) -> MarketState:
    """MarketAgent 실행"""
    print(f"  [MarketAgent] 시장 분석 시작...")
    agent = MarketAgent(company_name=company_name)
    result = agent.get_market_result()
    print(f"  [MarketAgent] 완료 - 점수: {result.get('competitor_score', 0)}점")
    return result


async def run_tech_agent(company_name: str) -> TechState:
    """TechAgent 실행"""
    print(f"  [TechAgent] 기술 분석 시작...")
    agent = TechAgent(startups_to_evaluate=company_name)
    result = agent.get_tech_result()
    print(f"  [TechAgent] 완료 - 점수: {result.get('technology_score', 0)}점")
    return result


async def run_comparison_agent(company_name: str) -> ComparisonState:
    """ComparisonAgent 실행"""
    print(f"  [ComparisonAgent] 경쟁사 비교 분석 시작...")
    agent = ComparisonAgent(company_name=company_name)
    result = agent.get_comparison_result()
    print(f"  [ComparisonAgent] 완료 - 점수: {result.get('competitor_score', 0)}점")
    return result


async def analyze_company_parallel(company_name: str) -> FinalState:
    """3개 Agent를 병렬로 실행하고 FinalState 생성"""
    print(f"\n[분석 시작] 3개 Agent 병렬 실행 중...\n")
    
    market_result, tech_result, comparison_result = await asyncio.gather(
        run_market_agent(company_name),
        run_tech_agent(company_name),
        run_comparison_agent(company_name),
        return_exceptions=True
    )
    
    # 에러 체크
    errors = []
    if isinstance(market_result, Exception):
        errors.append(f"MarketAgent: {str(market_result)}")
        market_result = None
    if isinstance(tech_result, Exception):
        errors.append(f"TechAgent: {str(tech_result)}")
        tech_result = None
    if isinstance(comparison_result, Exception):
        errors.append(f"ComparisonAgent: {str(comparison_result)}")
        comparison_result = None
    
    if errors:
        print(f"\n[경고] 일부 Agent 실행 실패:")
        for error in errors:
            print(f"  - {error}")
    
    # 모든 Agent가 실패한 경우에만 예외 발생
    if all(isinstance(r, Exception) for r in [market_result, tech_result, comparison_result]):
        error_msg = "\n  - ".join(errors)
        raise Exception(f"모든 Agent 실행 실패:\n  - {error_msg}")
    
    final_state: FinalState = {
        "company_name": company_name,
        "market": market_result,
        "tech": tech_result,
        "comparison": comparison_result
    }
    
    print(f"\n[분석 완료] FinalState 생성 완료")
    
    return final_state


# ===== LangGraph 노드 함수들 =====
def analyze_node(state: WorkflowState) -> WorkflowState:
    """분석 노드: 3개 Agent 병렬 실행"""
    try:
        company_name = state["company_name"]
        print(f"\n{'='*60}")
        print(f"[ANALYZE NODE] {company_name}")
        print(f"{'='*60}")
        
        # async 함수 실행
        final_state = asyncio.run(analyze_company_parallel(company_name))
        
        return {"final_state": final_state}
        
    except Exception as e:
        print(f"[ANALYZE NODE] 분석 실패: {str(e)}")
        return {"error": str(e)}


def judge_node(state: WorkflowState) -> WorkflowState:
    """판단 노드: 보고서 생성 여부 결정"""
    try:
        company_name = state["company_name"]
        final_state = state["final_state"]
        
        print(f"\n{'='*60}")
        print(f"[JUDGE NODE] {company_name}")
        print(f"{'='*60}")
        
        judge_agent = JudgeLLM(final_state=final_state)
        judgment = judge_agent.should_generate_report()
        
        # judge_result를 state에 임시 저장 (save_result_node에서 사용)
        state["_judge_agent_result"] = judge_agent.judge_result
        
        reason = judge_agent.get_rejection_reason() if not judgment else "모든 기준 통과"
        score = judge_agent.judge_result.get("score", 0.0) if judge_agent.judge_result else 0.0
        
        result_text = "ACCEPT ✓" if judgment else "REJECT ✗"
        print(f"\n판단 결과: {result_text}")
        print(f"평균 점수: {score:.1f}점")
        print(f"판단 이유: {reason}")
        
        return {
            "judgment": judgment,
            "_judge_agent_result": judge_agent.judge_result,
            "error": None
        }
        
    except Exception as e:
        print(f"[JUDGE NODE] 판단 실패: {str(e)}")
        return {"error": str(e)}


def save_result_node(state: WorkflowState) -> WorkflowState:
    """결과 저장 노드: Reject만 파일로 저장"""
    try:
        company_name = state["company_name"]
        final_state = state["final_state"]
        judgment = state.get("judgment", False)
        
        print(f"\n{'='*60}")
        print(f"[SAVE RESULT NODE] {company_name}")
        print(f"{'='*60}")
        
        # judge_node에서 전달받은 결과 사용
        judge_result = state.get("_judge_agent_result")
        
        if judge_result:
            reason = judge_result.get("reason", "이유 없음")
            score = judge_result.get("score", 0.0)
        else:
            # fallback: JudgeLLM 재실행
            judge_agent = JudgeLLM(final_state=final_state)
            reason = judge_agent.get_rejection_reason()
            score = judge_agent.judge_result.get("score", 0.0) if judge_agent.judge_result else 0.0
        
        # Reject인 경우만 파일 저장
        result_path = save_judgment_to_file(company_name, final_state, judgment, reason, score)
        
        print(f"\nREJECT 결과 저장 완료: {result_path}")
        
        return {"report_path": result_path}
        
    except Exception as e:
        print(f"[SAVE RESULT NODE] 저장 실패: {str(e)}")
        return {"error": str(e)}


def generate_report_node(state: WorkflowState) -> WorkflowState:
    """보고서 생성 노드: Accept인 경우 보고서 생성"""
    try:
        company_name = state["company_name"]
        final_state = state["final_state"]
        
        print(f"\n{'='*60}")
        print(f"[GENERATE REPORT NODE] {company_name}")
        print(f"{'='*60}")
        
        print(f"\n보고서 생성 중...")
        
        # ReportAgent로 보고서 생성
        report_agent = ReportLLM(final_state=final_state)
        report_content = report_agent.generate_report()
        
        # 파일 저장
        report_path = save_report_to_file(company_name, report_content)
        
        print(f"\n보고서 생성 완료: {report_path}")
        
        return {"report_path": report_path}
        
    except Exception as e:
        print(f"[GENERATE REPORT NODE] 보고서 생성 실패: {str(e)}")
        return {"error": str(e)}


# ===== Conditional Edge =====
def should_continue(state: WorkflowState) -> str:
    """다음 노드 결정"""
    if state.get("error"):
        return "end"
    
    judgment = state.get("judgment", False)
    
    # Accept이면 보고서 생성, Reject이면 결과만 저장
    return "generate_report" if judgment else "save_reject"


# ===== LangGraph 워크플로우 생성 =====
def create_workflow() -> StateGraph:
    """LangGraph 워크플로우 생성"""
    workflow = StateGraph(WorkflowState)
    
    # 노드 추가
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("save_reject", save_result_node)
    
    # 엣지 설정
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "judge")
    
    # 조건부 엣지: Accept -> 보고서 생성, Reject -> 결과 저장
    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {
            "generate_report": "generate_report",
            "save_reject": "save_reject",
            "end": END
        }
    )
    
    workflow.add_edge("generate_report", END)
    workflow.add_edge("save_reject", END)
    
    return workflow.compile()


# ===== 회사 리스트 처리 =====
def process_companies(companies: list[str]) -> dict:
    """여러 회사를 순차적으로 처리"""
    ensure_reports_dir()
    
    app = create_workflow()
    
    results = {
        "total": len(companies),
        "accepted": 0,
        "rejected": 0,
        "failed": 0,
        "details": {}
    }
    
    for idx, company_name in enumerate(companies, 1):
        print(f"\n\n{'#'*60}")
        print(f"# [{idx}/{len(companies)}] {company_name}")
        print(f"{'#'*60}")
        
        try:
            # 초기 상태
            initial_state: WorkflowState = {
                "company_name": company_name,
                "final_state": None,
                "judgment": None,
                "report_path": None,
                "error": None
            }
            
            # 워크플로우 실행
            final_workflow_state = app.invoke(initial_state)
            
            # 결과 분류
            if final_workflow_state.get("error"):
                results["failed"] += 1
                results["details"][company_name] = {
                    "status": "failed",
                    "error": final_workflow_state["error"]
                }
                
            elif final_workflow_state.get("judgment"):
                results["accepted"] += 1
                results["details"][company_name] = {
                    "status": "accepted",
                    "report_path": final_workflow_state.get("report_path")
                }
                
            else:
                results["rejected"] += 1
                results["details"][company_name] = {
                    "status": "rejected",
                    "report_path": final_workflow_state.get("report_path")
                }
                
        except Exception as e:
            results["failed"] += 1
            results["details"][company_name] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"\n[오류] {company_name} 처리 실패: {str(e)}")
    
    return results


# ===== 최종 결과 출력 =====
def print_final_summary(results: dict):
    """최종 결과 요약 출력"""
    print("\n\n" + "="*60)
    print("최종 결과 요약")
    print("="*60)
    print(f"\n전체: {results['total']}개")
    print(f"수락(Accept): {results['accepted']}개")
    print(f"거부(Reject): {results['rejected']}개")
    print(f"실패(Error): {results['failed']}개")
    
    print("\n" + "="*60)
    print("개별 결과")
    print("="*60)
    
    for company, detail in results["details"].items():
        if detail["status"] == "failed":
            print(f"\n❌ {company} - ERROR")
            print(f"   오류: {detail['error']}")
        elif detail["status"] == "accepted":
            print(f"\n✓ {company} - ACCEPT")
            print(f"   파일: {detail.get('report_path', 'N/A')}")
        else:
            print(f"\n✗ {company} - REJECT")
            print(f"   파일: {detail.get('report_path', 'N/A')}")
    
    print("\n")


# ===== Main 함수 =====
def main():
    """메인 함수"""
    print("\n" + "="*60)
    print("기업 분석 및 판단 시스템 (LangGraph)")
    print("="*60)
    print(f"\n처리할 회사: {len(COMPANIES)}개")
    for idx, company in enumerate(COMPANIES, 1):
        print(f"  {idx}. {company}")
    print(f"\n기준 점수: 평균 75점 이상")
    print()
    
    try:
        # 회사 리스트 처리
        results = process_companies(COMPANIES)
        
        # 최종 결과 출력
        print_final_summary(results)
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n치명적 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    result = main()