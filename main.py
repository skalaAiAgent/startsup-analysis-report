import os
import sys
import asyncio
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# Agent import
from agents.market_agent import MarketAgent
from agents.tech_agent import TechAgent
from agents.comparison_agent import ComparisonAgent

# State import
from states.market_state import MarketState
from states.tech_state import TechState
from states.comparison_state import ComparisonState
from states.final_state import FinalState


# ===== 병렬 분석 실행 =====

async def analyze_company_parallel(company_name: str) -> FinalState:
    """
    3개 Agent를 병렬로 실행하고 FinalState 생성
    
    Args:
        company_name: 분석할 회사명
        
    Returns:
        FinalState: 통합 분석 결과
        
    Raises:
        Exception: Agent 실행 중 에러 발생 시
    """
    print(f"\n{'='*60}")
    print(f"[{company_name}] 분석 시작")
    print(f"{'='*60}\n")
    
    try:
        # 3개 Agent 병렬 실행
        print("3개 Agent 병렬 실행 중...\n")
        
        market_result, tech_result, comparison_result = await asyncio.gather(
            run_market_agent(company_name),
            run_tech_agent(company_name),
            run_comparison_agent(company_name),
            return_exceptions=True  # 개별 에러를 반환하여 처리
        )
        
        # 에러 체크
        errors = []
        if isinstance(market_result, Exception):
            errors.append(f"MarketAgent: {str(market_result)}")
        if isinstance(tech_result, Exception):
            errors.append(f"TechAgent: {str(tech_result)}")
        if isinstance(comparison_result, Exception):
            errors.append(f"ComparisonAgent: {str(comparison_result)}")
        
        if errors:
            error_msg = "\n  - ".join(errors)
            raise Exception(f"Agent 실행 중 에러 발생:\n  - {error_msg}")
        
        # FinalState 생성
        print(f"\n{'='*60}")
        print("모든 분석 완료! FinalState 생성 중...")
        print(f"{'='*60}\n")
        
        final_state: FinalState = {
            "company_name": company_name,
            "market": market_result,
            "tech": tech_result,
            "comparison": comparison_result
        }
        
        return final_state
        
    except Exception as e:
        print(f"\n에러가 발생하여 분석을 완료할 수 없습니다.")
        print(f"상세: {str(e)}\n")
        raise


async def run_market_agent(company_name: str) -> MarketState:
    """MarketAgent 실행"""
    print(f"  [MarketAgent] 시장 분석 시작...")
    try:
        agent = MarketAgent(company_name=company_name)
        result = agent.get_market_result()
        print(f"  [MarketAgent] 완료")
        return result
    except Exception as e:
        print(f"  [MarketAgent] 실패: {str(e)}")
        raise


async def run_tech_agent(company_name: str) -> TechState:
    """TechAgent 실행"""
    print(f"  [TechAgent] 기술 분석 시작...")
    try:
        agent = TechAgent(company_name=company_name)
        result = agent.get_tech_result()
        print(f"  [TechAgent] 완료")
        return result
    except Exception as e:
        print(f"  [TechAgent] 실패: {str(e)}")
        raise


async def run_comparison_agent(company_name: str) -> ComparisonState:
    """ComparisonAgent 실행"""
    print(f"  [ComparisonAgent] 경쟁사 비교 분석 시작...")
    try:
        agent = ComparisonAgent(company_name=company_name)
        result = agent.get_comparison_result()
        print(f"  [ComparisonAgent] 완료")
        return result
    except Exception as e:
        print(f"  [ComparisonAgent] 실패: {str(e)}")
        raise


# ===== Main 함수 =====

def main():
    """LangGraph 워크플로우 메인 함수"""
    
    # 회사명 입력 받기
    if len(sys.argv) > 1:
        company_name = sys.argv[1]
    else:
        company_name = input("분석할 회사명을 입력하세요: ").strip()
        if not company_name:
            print("회사명을 입력해주세요.")
            sys.exit(1)
    
    try:
        # 비동기 실행
        result = asyncio.run(analyze_company_parallel(company_name))
        
        # 결과 출력
        print("\n" + "="*60)
        print("FinalState 생성 완료!")
        print("="*60)
        print(f"\n회사명: {result['company_name']}")
        print(f"\nMarketState: {type(result['market']).__name__}")
        print(f"TechState: {type(result['tech']).__name__}")
        print(f"ComparisonState: {type(result['comparison']).__name__}")
        print()
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n치명적 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    result = main()