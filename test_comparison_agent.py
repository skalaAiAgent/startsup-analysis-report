import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# LangGraph import
from langgraph.graph import StateGraph, END

# Agent import - ComparisonAgent와 JudgeAgent만 사용
from agents.company_comparison_agent import ComparisonAgent
from llm.judge_llm import JudgeLLM

# State import
from state.comparison_state import ComparisonState
from state.final_state import FinalState
from state.workflow_state import WorkflowState


# ===== 상수 정의 =====
TEST_COMPANIES = ["트립소다"]  # 테스트할 회사 (필요시 추가 가능)
RESULTS_DIR = "results/comparison_judge_test"  # 테스트용 결과 디렉토리


# ===== 유틸리티 함수 =====
def ensure_results_dir():
    """results 디렉토리 생성"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"[INFO] {RESULTS_DIR} 디렉토리 생성됨")


def get_timestamp():
    """타임스탬프 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_judge_result_to_file(company_name: str, judgment: bool, reason: str = None) -> str:
    """Judge 결과를 파일로 저장"""
    timestamp = get_timestamp()
    status = "ACCEPT" if judgment else "REJECT"
    filename = f"{company_name}_judge_{status}_{timestamp}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    content = f"""회사명: {company_name}
분석 일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
판단 결과: {status}

[ComparisonAgent + JudgeAgent 테스트]
- Market State: None (테스트에서 스킵)
- Tech State: None (테스트에서 스킵)
- Comparison State: 실행됨
"""
    
    if not judgment and reason:
        content += f"\nReject 이유:\n{reason}"
    
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
    print(f"ComparisonAgent 실행 중...")
    
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


# ===== LangGraph 노드 함수들 (Judge까지만) =====
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
    """판단 노드: 보고서 생성 여부 결정 (여기서 종료)"""
    try:
        company_name = state["company_name"]
        final_state = state["final_state"]
        
        print(f"[JUDGE] {company_name} 보고서 생성 여부 판단 중...")
        print(f"[JUDGE] (참고: market=None, tech=None 상태)")
        
        judge_agent = JudgeLLM(final_state=final_state)
        judgment = judge_agent.should_generate_report()
        
        result = "ACCEPT" if judgment else "REJECT"
        print(f"[JUDGE] {company_name} 판단 결과: {result}")
        
        # reject인 경우 이유도 가져오기
        rejection_reason = None
        if not judgment:
            rejection_reason = judge_agent.get_rejection_reason()
            print(f"[JUDGE] Reject 이유: {rejection_reason}")
        
        # 결과를 파일로 저장
        result_path = save_judge_result_to_file(company_name, judgment, rejection_reason)
        print(f"[JUDGE] 결과 저장 완료: {result_path}")
        
        return {
            "judgment": judgment,
            "rejection_reason": rejection_reason,
            "result_path": result_path
        }
        
    except Exception as e:
        print(f"[JUDGE] {company_name} 판단 실패: {str(e)}")
        return {"error": str(e)}


# ===== LangGraph 워크플로우 생성 (Judge까지만) =====
def create_workflow_judge_only() -> StateGraph:
    """LangGraph 워크플로우 생성 (Judge까지만)"""
    workflow = StateGraph(WorkflowState)
    
    # 노드 추가 - analyze와 judge만
    workflow.add_node("analyze", analyze_node_comparison_only)
    workflow.add_node("judge", judge_node)
    
    # 엣지 설정 - judge 이후 바로 종료
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "judge")
    workflow.add_edge("judge", END)
    
    return workflow.compile()


# ===== 회사 리스트 처리 =====
def process_companies_judge_only(companies: list[str]) -> dict:
    """여러 회사를 순차적으로 처리 (Judge까지만)"""
    ensure_results_dir()
    
    app = create_workflow_judge_only()
    
    results = {
        "total": len(companies),
        "accepted": 0,
        "rejected": 0,
        "failed": 0,
        "details": {}
    }
    
    for idx, company_name in enumerate(companies, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(companies)}] {company_name} 처리 시작 (Judge까지만)")
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
                results["accepted"] += 1
                results["details"][company_name] = {
                    "status": "accepted",
                    "result_path": final_state.get("result_path")
                }
                print(f"\n[RESULT] {company_name}: ACCEPTED")
                
            else:
                results["rejected"] += 1
                results["details"][company_name] = {
                    "status": "rejected",
                    "result_path": final_state.get("result_path"),
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
    """메인 함수 (ComparisonAgent + JudgeAgent 테스트)"""
    print("\n" + "="*60)
    print("ComparisonAgent + JudgeAgent 테스트")
    print("Judge 판단까지만 실행 (보고서 생성 안 함)")
    print("="*60)
    print(f"\n처리할 회사: {len(TEST_COMPANIES)}개")
    for idx, company in enumerate(TEST_COMPANIES, 1):
        print(f"  {idx}. {company}")
    print("\n[주의] MarketAgent, TechAgent는 스킵됩니다.")
    print("[주의] 보고서는 생성되지 않고 Judge 결과만 저장됩니다.")
    print()
    
    try:
        # 회사 리스트 처리
        results = process_companies_judge_only(TEST_COMPANIES)
        
        # 최종 결과 출력
        print("\n" + "="*60)
        print("Judge 테스트 최종 결과")
        print("="*60)
        print(f"\n전체: {results['total']}개")
        print(f"Accept: {results['accepted']}개")
        print(f"Reject: {results['rejected']}개")
        print(f"실패: {results['failed']}개")
        
        print("\n상세:")
        for company, detail in results["details"].items():
            status = detail["status"]
            if status == "accepted":
                print(f"  [ACCEPTED] {company}")
                print(f"    결과 파일: {detail.get('result_path')}")
            elif status == "rejected":
                print(f"  [REJECTED] {company}")
                print(f"    결과 파일: {detail.get('result_path')}")
                print(f"    이유: {detail.get('reason')}")
            else:
                print(f"  [FAILED] {company}")
                print(f"    에러: {detail['error']}")
        
        print(f"\n결과 저장 위치: {RESULTS_DIR}/")
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