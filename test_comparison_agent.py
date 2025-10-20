import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# LangGraph import
from langgraph.graph import StateGraph, END

# Agent import - ComparisonAgent만 사용
from agents.company_comparison_agent import ComparisonAgent
from agents.judge_agent import JudgeAgent
from agents.report_agent import ReportAgent

# State import
from state.comparison_state import ComparisonState
from state.final_state import FinalState
from state.workflow_state import WorkflowState


# ===== 상수 정의 =====
TEST_COMPANIES = ["트립소다"]  # 테스트할 회사 (필요시 추가 가능)
REPORTS_DIR = "reports/comparison_test"  # 테스트용 별도 디렉토리


# ===== 유틸리티 함수 =====
def ensure_reports_dir():
    """reports 디렉토리 생성"""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        print(f"[INFO] {REPORTS_DIR} 디렉토리 생성됨")


def get_timestamp():
    """타임스탬프 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_report_to_file(company_name: str, content: str) -> str:
    """보고서를 파일로 저장"""
    timestamp = get_timestamp()
    filename = f"{company_name}_comparison_only_report_{timestamp}.txt"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def save_rejection_to_file(company_name: str, final_state: FinalState, reason: str) -> str:
    """reject 이유를 파일로 저장"""
    timestamp = get_timestamp()
    filename = f"{company_name}_comparison_only_rejected_{timestamp}.txt"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    content = f"""회사명: {company_name}
분석 일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Reject 이유: {reason}

[ComparisonAgent만 실행한 테스트]
상세:
- Market State: None (테스트에서 스킵)
- Tech State: None (테스트에서 스킵)
- Comparison State: {type(final_state['comparison']).__name__}
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


# ===== ComparisonAgent만 실행 =====
async def run_comparison_agent_only(company_name: str) -> ComparisonState:
    """ComparisonAgent만 실행"""
    print(f"  [ComparisonAgent] 경쟁사 비교 분석 시작...")
    agent = ComparisonAgent(company_name=company_name)
    result = agent.get_comparison_result()
    print(f"  [ComparisonAgent] 완료")
    return result


async def analyze_company_comparison_only(company_name: str) -> FinalState:
    """ComparisonAgent만 실행하고 FinalState 생성 (market, tech는 None)"""
    print(f"ComparisonAgent만 실행 중...")
    
    try:
        comparison_result = await run_comparison_agent_only(company_name)
    except Exception as e:
        raise Exception(f"ComparisonAgent 실행 중 에러 발생: {str(e)}")
    
    final_state: FinalState = {
        "company_name": company_name,
        "market": None,  # MarketAgent 스킵
        "tech": None,  # TechAgent 스킵
        "comparison": comparison_result
    }
    
    return final_state


# ===== LangGraph 노드 함수들 (ComparisonAgent 전용) =====
def analyze_node_comparison_only(state: WorkflowState) -> WorkflowState:
    """분석 노드: ComparisonAgent만 실행"""
    try:
        company_name = state["company_name"]
        print(f"\n[ANALYZE - COMPARISON ONLY] {company_name} 분석 시작")
        
        # async 함수 실행
        final_state = asyncio.run(analyze_company_comparison_only(company_name))
        
        print(f"[ANALYZE - COMPARISON ONLY] {company_name} 분석 완료")
        return {"final_state": final_state}
        
    except Exception as e:
        print(f"[ANALYZE - COMPARISON ONLY] {company_name} 분석 실패: {str(e)}")
        return {"error": str(e)}


def judge_node(state: WorkflowState) -> WorkflowState:
    """판단 노드: 보고서 생성 여부 결정"""
    try:
        company_name = state["company_name"]
        final_state = state["final_state"]
        
        print(f"[JUDGE] {company_name} 보고서 생성 여부 판단 중...")
        print(f"[JUDGE] (참고: market=None, tech=None 상태)")
        
        judge_agent = JudgeAgent(final_state=final_state)
        judgment = judge_agent.should_generate_report()
        
        result = "accept" if judgment else "reject"
        print(f"[JUDGE] {company_name} 판단 결과: {result}")
        
        # reject인 경우 이유도 가져오기
        rejection_reason = None
        if not judgment:
            rejection_reason = judge_agent.get_rejection_reason()
        
        return {
            "judgment": judgment,
            "rejection_reason": rejection_reason
        }
        
    except Exception as e:
        print(f"[JUDGE] {company_name} 판단 실패: {str(e)}")
        return {"error": str(e)}


def generate_report_node(state: WorkflowState) -> WorkflowState:
    """보고서 생성 노드"""
    try:
        company_name = state["company_name"]
        final_state = state["final_state"]
        
        print(f"[REPORT] {company_name} 보고서 생성 중...")
        print(f"[REPORT] (참고: Comparison 분석 결과만 사용)")
        
        report_agent = ReportAgent(final_state=final_state)
        report_content = report_agent.generate_report()
        
        # 파일로 저장
        report_path = save_report_to_file(company_name, report_content)
        
        print(f"[REPORT] {company_name} 보고서 생성 완료: {report_path}")
        return {"report_path": report_path}
        
    except Exception as e:
        print(f"[REPORT] {company_name} 보고서 생성 실패: {str(e)}")
        return {"error": str(e)}


def save_rejection_node(state: WorkflowState) -> WorkflowState:
    """Reject 이유 저장 노드"""
    try:
        company_name = state["company_name"]
        final_state = state["final_state"]
        rejection_reason = state.get("rejection_reason", "이유 없음")
        
        print(f"[REJECT] {company_name} reject 이유 저장 중...")
        
        rejection_path = save_rejection_to_file(company_name, final_state, rejection_reason)
        
        print(f"[REJECT] {company_name} reject 이유 저장 완료: {rejection_path}")
        return {"report_path": rejection_path}
        
    except Exception as e:
        print(f"[REJECT] {company_name} 저장 실패: {str(e)}")
        return {"error": str(e)}


# ===== Conditional Edge =====
def should_generate_report(state: WorkflowState) -> str:
    """보고서 생성 여부에 따라 다음 노드 결정"""
    if state.get("error"):
        return "end"
    
    judgment = state.get("judgment", False)
    return "generate" if judgment else "reject"


# ===== LangGraph 워크플로우 생성 (ComparisonAgent 전용) =====
def create_workflow_comparison_only() -> StateGraph:
    """LangGraph 워크플로우 생성 (ComparisonAgent 전용)"""
    workflow = StateGraph(WorkflowState)
    
    # 노드 추가 - analyze 노드만 comparison 전용으로 변경
    workflow.add_node("analyze", analyze_node_comparison_only)
    workflow.add_node("judge", judge_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("save_rejection", save_rejection_node)
    
    # 엣지 설정 (동일)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "judge")
    
    # 조건부 엣지
    workflow.add_conditional_edges(
        "judge",
        should_generate_report,
        {
            "generate": "generate_report",
            "reject": "save_rejection",
            "end": END
        }
    )
    
    workflow.add_edge("generate_report", END)
    workflow.add_edge("save_rejection", END)
    
    return workflow.compile()


# ===== 회사 리스트 처리 =====
def process_companies_comparison_only(companies: list[str]) -> dict:
    """여러 회사를 순차적으로 처리 (ComparisonAgent만)"""
    ensure_reports_dir()
    
    app = create_workflow_comparison_only()
    
    results = {
        "total": len(companies),
        "success": 0,
        "rejected": 0,
        "failed": 0,
        "details": {}
    }
    
    for idx, company_name in enumerate(companies, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(companies)}] {company_name} 처리 시작 (ComparisonAgent만)")
        print(f"{'='*60}")
        
        try:
            # 초기 상태
            initial_state: WorkflowState = {
                "company_name": company_name,
                "final_state": None,
                "judgment": None,
                "report_path": None,
                "rejection_reason": None,
                "error": None
            }
            
            # 워크플로우 실행
            final_state = app.invoke(initial_state)
            
            # 결과 분류
            if final_state.get("error"):
                results["failed"] += 1
                results["details"][company_name] = {
                    "status": "failed",
                    "error": final_state["error"]
                }
                print(f"\n[RESULT] {company_name}: FAILED")
                
            elif final_state.get("judgment"):
                results["success"] += 1
                results["details"][company_name] = {
                    "status": "success",
                    "report_path": final_state["report_path"]
                }
                print(f"\n[RESULT] {company_name}: SUCCESS")
                
            else:
                results["rejected"] += 1
                results["details"][company_name] = {
                    "status": "rejected",
                    "rejection_path": final_state["report_path"],
                    "reason": final_state.get("rejection_reason")
                }
                print(f"\n[RESULT] {company_name}: REJECTED")
                
        except Exception as e:
            results["failed"] += 1
            results["details"][company_name] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"\n[RESULT] {company_name}: FAILED - {str(e)}")
    
    return results


# ===== Main 함수 =====
def main():
    """메인 함수 (ComparisonAgent 테스트)"""
    print("\n" + "="*60)
    print("ComparisonAgent 테스트 - 경쟁사 비교 분석만 수행")
    print("="*60)
    print(f"\n처리할 회사: {len(TEST_COMPANIES)}개")
    for idx, company in enumerate(TEST_COMPANIES, 1):
        print(f"  {idx}. {company}")
    print("\n[주의] MarketAgent, TechAgent는 스킵됩니다.")
    print()
    
    try:
        # 회사 리스트 처리
        results = process_companies_comparison_only(TEST_COMPANIES)
        
        # 최종 결과 출력
        print("\n" + "="*60)
        print("ComparisonAgent 테스트 최종 결과")
        print("="*60)
        print(f"\n전체: {results['total']}개")
        print(f"성공: {results['success']}개")
        print(f"거부: {results['rejected']}개")
        print(f"실패: {results['failed']}개")
        
        print("\n상세:")
        for company, detail in results["details"].items():
            status = detail["status"]
            if status == "success":
                print(f"  [{status.upper()}] {company}")
                print(f"    보고서: {detail['report_path']}")
            elif status == "rejected":
                print(f"  [{status.upper()}] {company}")
                print(f"    파일: {detail['rejection_path']}")
                print(f"    이유: {detail['reason']}")
            else:
                print(f"  [{status.upper()}] {company}")
                print(f"    에러: {detail['error']}")
        
        print(f"\n보고서 저장 위치: {REPORTS_DIR}/")
        print()
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