import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# Agent import
from agents.tech_agent import TechAgent
from llm.judge_llm import JudgeLLM
from llm.report_llm import ReportLLM

# State import
from state.tech_state import TechState
from state.final_state import FinalState


# ===== 상수 정의 =====
TEST_COMPANIES = ["트립소다"]  # 테스트할 회사 (필요시 추가 가능)
REPORTS_DIR = "reports/tech_test"  # 테스트용 별도 디렉토리


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
    filename = f"{company_name}_tech_only_report_{timestamp}.txt"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def save_rejection_to_file(company_name: str, tech_state: TechState, reason: str) -> str:
    """reject 이유를 파일로 저장"""
    timestamp = get_timestamp()
    filename = f"{company_name}_tech_only_rejected_{timestamp}.txt"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    content = f"""회사명: {company_name}
분석 일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Reject 이유: {reason}

[TechAgent 분석 결과]
기술 점수: {tech_state['technology_score']}/100

항목별 점수:
- 혁신성(innovation): {tech_state['category_scores']['innovation']}/30
- 완성도(completeness): {tech_state['category_scores']['completeness']}/30
- 경쟁력(competitiveness): {tech_state['category_scores']['competitiveness']}/20
- 특허(patent): {tech_state['category_scores']['patent']}/10
- 확장성(scalability): {tech_state['category_scores']['scalability']}/10

분석 근거:
{tech_state['technology_analysis_basis']}
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def print_tech_state(tech_state: TechState):
    """TechState 내용 출력"""
    print(f"\n{'='*60}")
    print(f"[TechState 결과]")
    print(f"{'='*60}")
    print(f"회사명: {tech_state['company_name']}")
    print(f"기술 점수: {tech_state['technology_score']}/100")
    print(f"\n항목별 점수:")
    print(f"  혁신성(innovation): {tech_state['category_scores']['innovation']}/30")
    print(f"  완성도(completeness): {tech_state['category_scores']['completeness']}/30")
    print(f"  경쟁력(competitiveness): {tech_state['category_scores']['competitiveness']}/20")
    print(f"  특허(patent): {tech_state['category_scores']['patent']}/10")
    print(f"  확장성(scalability): {tech_state['category_scores']['scalability']}/10")
    print(f"\n분석 근거:")
    print(f"{tech_state['technology_analysis_basis'][:200]}...")
    print(f"{'='*60}\n")


# ===== TechAgent 실행 =====
def run_tech_agent(company_name: str) -> TechState:
    """TechAgent 실행"""
    print(f"\n[TechAgent] 기술 분석 시작: {company_name}")
    agent = TechAgent(startups_to_evaluate=company_name)
    result = agent.get_tech_result()
    print(f"[TechAgent] 완료")
    return result


def analyze_company_tech_only(company_name: str) -> FinalState:
    """TechAgent만 실행하고 FinalState 생성"""
    print(f"\n{'='*60}")
    print(f"[분석 시작] {company_name}")
    print(f"{'='*60}")
    
    try:
        tech_result = run_tech_agent(company_name)
        print_tech_state(tech_result)
    except Exception as e:
        raise Exception(f"TechAgent 실행 중 에러 발생: {str(e)}")
    
    # FinalState 생성 (market, comparison은 None)
    final_state: FinalState = {
        "company_name": company_name,
        "market": None,
        "tech": tech_result,
        "comparison": None
    }
    
    return final_state


# ===== Judge & Report 실행 =====
def judge_and_report(final_state: FinalState) -> dict:
    """판단 및 보고서 생성"""
    company_name = final_state["company_name"]
    
    # 1. JudgeAgent 실행
    print(f"\n[JUDGE] {company_name} 보고서 생성 여부 판단 중...")
    judge_agent = JudgeLLM(final_state=final_state)
    judgment = judge_agent.should_generate_report()
    
    result = {
        "company_name": company_name,
        "judgment": judgment,
        "file_path": None,
        "rejection_reason": None
    }
    
    # 2. Accept인 경우 보고서 생성
    if judgment:
        print(f"[JUDGE] {company_name} 판단 결과: ACCEPT (보고서 생성)")
        
        print(f"\n[REPORT] {company_name} 보고서 생성 중...")
        report_agent = ReportLLM(final_state=final_state)
        report_content = report_agent.generate_report()
        
        # 파일 저장
        report_path = save_report_to_file(company_name, report_content)
        result["file_path"] = report_path
        
        print(f"[REPORT] {company_name} 보고서 생성 완료")
        print(f"  파일: {report_path}")
    
    # 3. Reject인 경우 이유 저장
    else:
        print(f"[JUDGE] {company_name} 판단 결과: REJECT (보고서 생성 안 함)")
        
        rejection_reason = judge_agent.get_rejection_reason()
        result["rejection_reason"] = rejection_reason
        
        print(f"\n[REJECT] {company_name} reject 이유 저장 중...")
        print(f"  이유: {rejection_reason}")
        
        # 파일 저장
        rejection_path = save_rejection_to_file(
            company_name, 
            final_state["tech"], 
            rejection_reason
        )
        result["file_path"] = rejection_path
        
        print(f"[REJECT] {company_name} reject 이유 저장 완료")
        print(f"  파일: {rejection_path}")
    
    return result


# ===== 회사 리스트 처리 =====
def process_companies_tech_only(companies: list[str]) -> dict:
    """여러 회사를 순차적으로 처리"""
    ensure_reports_dir()
    
    results = {
        "total": len(companies),
        "success": 0,
        "rejected": 0,
        "failed": 0,
        "details": {}
    }
    
    for idx, company_name in enumerate(companies, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(companies)}] {company_name} 처리 시작")
        print(f"{'='*80}")
        
        try:
            # 1. TechAgent 실행
            final_state = analyze_company_tech_only(company_name)
            
            # 2. Judge & Report 실행
            result = judge_and_report(final_state)
            
            # 3. 결과 분류
            if result["judgment"]:
                results["success"] += 1
                results["details"][company_name] = {
                    "status": "success",
                    "report_path": result["file_path"]
                }
                print(f"\n[최종 결과] {company_name}: SUCCESS ✓")
            else:
                results["rejected"] += 1
                results["details"][company_name] = {
                    "status": "rejected",
                    "rejection_path": result["file_path"],
                    "reason": result["rejection_reason"]
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
    print("TechAgent 테스트 - 기술 분석 및 보고서 생성 여부 판단")
    print("="*80)
    print(f"\n처리할 회사: {len(TEST_COMPANIES)}개")
    for idx, company in enumerate(TEST_COMPANIES, 1):
        print(f"  {idx}. {company}")
    print("\n[주의] MarketAgent, ComparisonAgent는 스킵됩니다.")
    print()
    
    try:
        # 회사 리스트 처리
        results = process_companies_tech_only(TEST_COMPANIES)
        
        # 최종 결과 출력
        print("\n" + "="*80)
        print("TechAgent 테스트 최종 결과")
        print("="*80)
        print(f"\n전체: {results['total']}개")
        print(f"성공 (보고서 생성): {results['success']}개")
        print(f"거부 (보고서 미생성): {results['rejected']}개")
        print(f"실패 (에러 발생): {results['failed']}개")
        
        print("\n" + "-"*80)
        print("상세 결과:")
        print("-"*80)
        for company, detail in results["details"].items():
            status = detail["status"]
            if status == "success":
                print(f"\n✓ [SUCCESS] {company}")
                print(f"  보고서: {detail['report_path']}")
            elif status == "rejected":
                print(f"\n✗ [REJECTED] {company}")
                print(f"  파일: {detail['rejection_path']}")
                print(f"  이유: {detail['reason']}")
            else:
                print(f"\n✗ [FAILED] {company}")
                print(f"  에러: {detail['error']}")
        
        print(f"\n{'='*80}")
        print(f"보고서 저장 위치: {REPORTS_DIR}/")
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