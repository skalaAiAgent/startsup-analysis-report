import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# Agent import
from agents.market_evaluation_agent import MarketAgent
from agents.judge_agent import JudgeAgent

# State import
from state.market_state import MarketState
from state.final_state import FinalState


# ===== 상수 정의 =====
TEST_COMPANIES = ["트립소다"]  # 테스트할 회사
REPORTS_DIR = "reports/market_test"  # 테스트용 디렉토리


# ===== 유틸리티 함수 =====
def ensure_reports_dir():
    """reports 디렉토리 생성"""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        print(f"[INFO] {REPORTS_DIR} 디렉토리 생성됨")


def get_timestamp():
    """타임스탬프 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_result_to_file(company_name: str, market_state: MarketState, judgment: bool, reason: str = None) -> str:
    """결과를 파일로 저장"""
    timestamp = get_timestamp()
    status = "ACCEPT" if judgment else "REJECT"
    filename = f"{company_name}_market_{status}_{timestamp}.txt"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    content = f"""회사명: {company_name}
분석 일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
판단 결과: {status}

[MarketAgent 분석 결과]
시장 점수: {market_state['competitor_score']}/100

분석 근거:
{market_state['competitor_analysis_basis']}
"""
    
    if not judgment and reason:
        content += f"\n\n[Reject 이유]\n{reason}"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def print_market_state(market_state: MarketState):
    """MarketState 내용 출력"""
    print(f"\n{'='*60}")
    print(f"[MarketState 결과]")
    print(f"{'='*60}")
    print(f"회사명: {market_state['company_name']}")
    print(f"시장 점수: {market_state['competitor_score']}/100")
    print(f"\n분석 근거:")
    print(f"{market_state['competitor_analysis_basis']}")
    print(f"{'='*60}\n")


# ===== MarketAgent 실행 =====
def run_market_agent(company_name: str) -> MarketState:
    """MarketAgent 실행"""
    print(f"\n[MarketAgent] 시장 분석 시작: {company_name}")
    agent = MarketAgent(company_name=company_name)
    result = agent.get_market_result()
    print(f"[MarketAgent] 완료")
    return result


def analyze_company_market_only(company_name: str) -> FinalState:
    """MarketAgent만 실행하고 FinalState 생성"""
    print(f"\n{'='*60}")
    print(f"[분석 시작] {company_name}")
    print(f"{'='*60}")
    
    try:
        market_result = run_market_agent(company_name)
        print_market_state(market_result)
    except Exception as e:
        raise Exception(f"MarketAgent 실행 중 에러 발생: {str(e)}")
    
    # FinalState 생성 (tech, comparison은 None)
    final_state: FinalState = {
        "company_name": company_name,
        "market": market_result,
        "tech": None,
        "comparison": None
    }
    
    return final_state


# ===== Judge 실행 =====
def judge_and_save(final_state: FinalState) -> dict:
    """판단 및 결과 저장"""
    company_name = final_state["company_name"]
    
    # 1. JudgeAgent 실행
    print(f"\n[JUDGE] {company_name} 보고서 생성 여부 판단 중...")
    judge_agent = JudgeAgent(final_state=final_state)
    judgment = judge_agent.should_generate_report()
    
    result = {
        "company_name": company_name,
        "judgment": judgment,
        "file_path": None,
        "rejection_reason": None
    }
    
    # 2. 결과에 따라 처리
    if judgment:
        print(f"[JUDGE] {company_name} 판단 결과: ACCEPT")
    else:
        print(f"[JUDGE] {company_name} 판단 결과: REJECT")
        rejection_reason = judge_agent.get_rejection_reason()
        result["rejection_reason"] = rejection_reason
        print(f"  이유: {rejection_reason}")
    
    # 3. 파일 저장
    print(f"\n[SAVE] {company_name} 결과 저장 중...")
    file_path = save_result_to_file(
        company_name,
        final_state["market"],
        judgment,
        result.get("rejection_reason")
    )
    result["file_path"] = file_path
    print(f"[SAVE] 저장 완료: {file_path}")
    
    return result


# ===== 회사 리스트 처리 =====
def process_companies_market_only(companies: list[str]) -> dict:
    """여러 회사를 순차적으로 처리"""
    ensure_reports_dir()
    
    results = {
        "total": len(companies),
        "accepted": 0,
        "rejected": 0,
        "failed": 0,
        "details": {}
    }
    
    for idx, company_name in enumerate(companies, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(companies)}] {company_name} 처리 시작")
        print(f"{'='*80}")
        
        try:
            # 1. MarketAgent 실행
            final_state = analyze_company_market_only(company_name)
            
            # 2. Judge 실행
            result = judge_and_save(final_state)
            
            # 3. 결과 분류
            if result["judgment"]:
                results["accepted"] += 1
                results["details"][company_name] = {
                    "status": "accepted",
                    "file_path": result["file_path"],
                    "market_score": final_state["market"]["competitor_score"]
                }
                print(f"\n[최종 결과] {company_name}: ACCEPTED ✓")
            else:
                results["rejected"] += 1
                results["details"][company_name] = {
                    "status": "rejected",
                    "file_path": result["file_path"],
                    "reason": result["rejection_reason"],
                    "market_score": final_state["market"]["competitor_score"]
                }
                print(f"\n[최종 결과] {company_name}: REJECTED ✗")
                
        except Exception as e:
            results["failed"] += 1
            results["details"][company_name] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"\n[최종 결과] {company_name}: FAILED - {str(e)}")
            import traceback
            traceback.print_exc()
    
    return results


# ===== Main 함수 =====
def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("MarketAgent 테스트 - 시장 분석 및 보고서 생성 여부 판단")
    print("="*80)
    print(f"\n처리할 회사: {len(TEST_COMPANIES)}개")
    for idx, company in enumerate(TEST_COMPANIES, 1):
        print(f"  {idx}. {company}")
    print("\n[주의] TechAgent, ComparisonAgent는 스킵됩니다.")
    print()
    
    try:
        # 회사 리스트 처리
        results = process_companies_market_only(TEST_COMPANIES)
        
        # 최종 결과 출력
        print("\n" + "="*80)
        print("MarketAgent 테스트 최종 결과")
        print("="*80)
        print(f"\n전체: {results['total']}개")
        print(f"Accept: {results['accepted']}개")
        print(f"Reject: {results['rejected']}개")
        print(f"실패: {results['failed']}개")
        
        print("\n" + "-"*80)
        print("상세 결과:")
        print("-"*80)
        for company, detail in results["details"].items():
            status = detail["status"]
            if status == "accepted":
                print(f"\n✓ [ACCEPTED] {company}")
                print(f"  시장 점수: {detail['market_score']}/100")
                print(f"  파일: {detail['file_path']}")
            elif status == "rejected":
                print(f"\n✗ [REJECTED] {company}")
                print(f"  시장 점수: {detail['market_score']}/100")
                print(f"  파일: {detail['file_path']}")
                print(f"  이유: {detail['reason']}")
            else:
                print(f"\n✗ [FAILED] {company}")
                print(f"  에러: {detail['error']}")
        
        print(f"\n{'='*80}")
        print(f"결과 저장 위치: {REPORTS_DIR}/")
        print(f"{'='*80}\n")
        
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