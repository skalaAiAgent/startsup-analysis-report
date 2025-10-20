import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# LangGraph import
from langgraph.graph import StateGraph, END

# Agent import - MarketEvaluator와 JudgeAgent만 사용
from agents.market_evaluation_agent import MarketEvaluator
from agents.judge_agent import JudgeAgent

# State import
from state.market_state import MarketState
from state.final_state import FinalState
from state.workflow_state import WorkflowState


# ===== 상수 정의 =====
TEST_COMPANIES = ["TripGenie AI"]  # 테스트할 회사 (필요시 추가 가능)
RESULTS_DIR = "results/market_judge_test"  # 테스트용 결과 디렉토리

# MarketEvaluator 설정
CHROMA_COLLECTION = "market_index"
CHROMA_PERSIST_DIR = "./rag/market/chroma"


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

[MarketEvaluator + JudgeAgent 테스트]
- Market State: 실행됨
- Tech State: None (테스트에서 스킵)
- Comparison State: None (테스트에서 스킵)
"""
    
    if not judgment and reason:
        content += f"\nReject 이유:\n{reason}"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


# ===== MarketEvaluator만 실행 =====
async def run_market_evaluator_only(company_name: str) -> MarketState:
    """MarketEvaluator만 실행"""
    print(f"  [MarketEvaluator] 시장 평가 분석 시작...")
    
    try:
        agent = MarketEvaluator(
            collection_name=CHROMA_COLLECTION,
            chroma_persist_dir=CHROMA_PERSIST_DIR,
            tavily_enabled=True
        )
        
        # evaluate_startup 실행 - 반환값: Dict[str, Any]
        raw_result = agent.evaluate_startup(company_name)
        
        # MarketState 형식으로 변환
        market_state: MarketState = {
            "company_name": raw_result.get("startup_name", company_name),
            "competitor_score": raw_result.get("market_score", 0),
            "competitor_analysis_basis": raw_result.get("rationale", "No rationale provided")
        }
        
        print(f"  [MarketEvaluator] 완료 - Score: {market_state['competitor_score']}")
        return market_state
        
    except Exception as e:
        print(f"  [MarketEvaluator] 에러 발생: {str(e)}")
        # 에러 발생 시 기본값 반환
        return {
            "company_name": company_name,
            "competitor_score": 0,
            "competitor_analysis_basis": f"Error during evaluation: {str(e)}"
        }


async def analyze_company_market_only(company_name: str) -> FinalState:
    """MarketEvaluator만 실행하고 FinalState 생성 (tech, comparison은 None)"""
    print(f"MarketEvaluator 실행 중...")
    
    try:
        market_result = await run_market_evaluator_only(company_name)
    except Exception as e:
        raise Exception(f"MarketEvaluator 실행 중 에러 발생: {str(e)}")
    
    final_state: FinalState = {
        "company_name": company_name,
        "market": market_result,
        "tech": None,  # TechAgent 스킵
        "comparison": None  # ComparisonAgent 스킵
    }
    
    return final_state


# ===== LangGraph 노드 함수들 (Judge까지만) =====
def analyze_node_market_only(state: WorkflowState) -> WorkflowState:
    """분석 노드: MarketEvaluator만 실행"""
    try:
        company_name = state["company_name"]
        print(f"\n[ANALYZE - MARKET ONLY] {company_name} 분석 시작")
        
        # async 함수 실행
        final_state = asyncio.run(analyze_company_market_only(company_name))
        
        print(f"[ANALYZE - MARKET ONLY] {company_name} 분석 완료")
        return {"final_state": final_state}
        
    except Exception as e:
        print(f"[ANALYZE - MARKET ONLY] {company_name} 분석 실패: {str(e)}")
        return {"error": str(e)}


def judge_node(state: WorkflowState) -> WorkflowState:
    """판단 노드: 보고서 생성 여부 결정 (여기서 종료)"""
    try:
        company_name = state["company_name"]
        final_state = state["final_state"]
        
        print(f"[JUDGE] {company_name} 보고서 생성 여부 판단 중...")
        print(f"[JUDGE] (참고: tech=None, comparison=None 상태)")
        
        judge_agent = JudgeAgent(final_state=final_state)
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
    workflow.add_node("analyze", analyze_node_market_only)
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
                    "result_path": final_state.get("result_path"),
                    "market_score": final_state.get("final_state", {}).get("market", {}).get("competitor_score")
                }
                print(f"\n[RESULT] {company_name}: ACCEPTED")
                
            else:
                results["rejected"] += 1
                results["details"][company_name] = {
                    "status": "rejected",
                    "result_path": final_state.get("result_path"),
                    "reason": final_state.get("rejection_reason"),
                    "market_score": final_state.get("final_state", {}).get("market", {}).get("competitor_score")
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
    """메인 함수 (MarketEvaluator + JudgeAgent 테스트)"""
    print("\n" + "="*60)
    print("MarketEvaluator + JudgeAgent 테스트")
    print("Judge 판단까지만 실행 (보고서 생성 안 함)")
    print("="*60)
    print(f"\n처리할 회사: {len(TEST_COMPANIES)}개")
    for idx, company in enumerate(TEST_COMPANIES, 1):
        print(f"  {idx}. {company}")
    print("\n[주의] TechAgent, ComparisonAgent는 스킵됩니다.")
    print("[주의] 보고서는 생성되지 않고 Judge 결과만 저장됩니다.")
    print(f"[설정] ChromaDB: {CHROMA_PERSIST_DIR}")
    print(f"[설정] Collection: {CHROMA_COLLECTION}")
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
                print(f"    Market Score: {detail.get('market_score', 'N/A')}")
                print(f"    결과 파일: {detail.get('result_path')}")
            elif status == "rejected":
                print(f"  [REJECTED] {company}")
                print(f"    Market Score: {detail.get('market_score', 'N/A')}")
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