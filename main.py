import os
import sys
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

from langgraph.graph import StateGraph, END
from agents.company_comparison_agent import analyze_competitor, AgentState


def main():
    """LangGraph 워크플로우 메인 함수"""
    
    print("="*70)
    print("🚀 Multi-Agent 스타트업 분석 시스템 시작")
    print("="*70)
    
    # 1. Graph 생성
    graph = StateGraph(AgentState)
    
    # 2. 노드 추가
    graph.add_node("AnalyzeCompetitor", analyze_competitor)
    # 향후 추가될 노드들:
    # graph.add_node("AnalyzeMarket", analyze_market)
    # graph.add_node("AnalyzeTech", analyze_tech)
    # graph.add_node("GenerateReport", generate_report)
    
    # 3. 엣지 연결
    graph.set_entry_point("AnalyzeCompetitor")
    graph.add_edge("AnalyzeCompetitor", END)
    
    # 향후 연결:
    # graph.add_edge("AnalyzeCompetitor", "AnalyzeMarket")
    # graph.add_edge("AnalyzeMarket", "AnalyzeTech")
    # graph.add_edge("AnalyzeTech", "GenerateReport")
    # graph.add_edge("GenerateReport", END)
    
    # 4. 컴파일
    app = graph.compile()
    
    # 5. 실행
    startup_name = os.environ.get("STARTUP_NAME", "어딩")
    print(f"\n📌 분석 대상: {startup_name}\n")
    
    result = app.invoke({"startup_name": startup_name})
    
    # 6. 결과 출력
    print("\n" + "="*70)
    print("📊 최종 분석 결과")
    print("="*70)
    print(f"\n🏢 기업명: {result.get('startup_name')}")
    print(f"🎯 경쟁사 점수: {result.get('competitor_score')}점")
    print(f"\n📋 분석 내용:")
    print("-"*70)
    print(result.get('competitor_analysis_basis', ''))
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = main()