from __future__ import annotations

import os
import sys
import re
from typing import Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# project root import path
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from state.comparison_state import ComparisonState
from rag.company_comparison.hybrid_store import HybridStore


def _extract_score_and_summary(text: str) -> tuple[int, str]:
    """LLM 출력에서 점수와 요약 추출"""
    score_patterns = [
        r"(?:score[:\s]*)?[\*]?(\d{1,3})[\*]?",
        r"\b(100|\d{1,2})/100",
        r"점수[:\s]*(\d{1,3})"
    ]
    score = 50
    
    for pattern in score_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            score = max(0, min(100, score))
            break
    
    # 점수 라인 제거
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines and re.match(r"^[\*\s]*\d{1,3}[\*\s]*$", lines[0]):
        lines = lines[1:]
    
    return score, "\n".join(lines)


class ComparisonAgent:
    """
    경쟁사 비교 분석 Agent
    
    PDF에 저장된 5개 회사의 데이터를 RAG로 검색하고,
    분석 대상 회사와 비교하여 점수와 분석 내용을 생성합니다.
    """

    def __init__(self, company_name: str) -> None:
        """
        ComparisonAgent 초기화
        
        Args:
            company_name: 분석 대상 회사명 (main.py에서 전달)
        """
        load_dotenv()
        self.company_name = company_name or "Unknown"
        
        # LLM 초기화
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        
        # RAG 디렉토리 설정
        rag_dir = os.path.join(
            os.path.dirname(__file__), 
            os.pardir, 
            "rag", 
            "company_comparison"
        )
        self.chroma_dir = os.path.join(rag_dir, ".chroma")
        
        # HybridStore 초기화
        self.store: HybridStore | None = None
        if os.path.exists(self.chroma_dir):
            try:
                self.store = HybridStore(
                    chroma_path=self.chroma_dir, 
                    collection="company_comparison"
                )
                print(f"✅ HybridStore 초기화 완료")
            except Exception as e:
                print(f"⚠️ HybridStore 초기화 실패: {e}")
                self.store = None
        else:
            print(f"⚠️ Chroma 인덱스가 없습니다: {self.chroma_dir}")

    def _build_context(self) -> str:
        """
        RAG에서 관련 컨텍스트 검색
        
        Returns:
            검색된 문서들의 컨텍스트 문자열
        """
        if not self.store:
            print("⚠️ HybridStore가 초기화되지 않음")
            return ""
        
        # 검색 쿼리: 현재 회사명 + 주요 키워드
        query = f"{self.company_name} 재무 매출 자산 부채 투자 MUV 소비자 세그먼트"
        print(f"🔎 검색 쿼리: {query}")
        
        try:
            # Hybrid search 실행
            results = self.store.search(
                query=query,
                k_bm25=15,
                k_vec=15,
                rrf_k=60,
                lambda_vec=0.3
            )
            
            if not results:
                print("⚠️ 검색 결과가 없습니다")
                return ""
            
            print(f"✅ {len(results)}개 문서 검색됨")
            
            # 중복 페이지 제거 및 상위 결과 선택
            seen = set()
            picked = []
            
            for r in results:
                page = (r.get("metadata") or {}).get("page")
                
                # 이미 본 페이지이고 충분히 모았으면 스킵
                if page in seen and len(picked) >= 12:
                    continue
                
                seen.add(page)
                picked.append(r)
                
                if len(picked) >= 20:
                    break
            
            print(f"📄 {len(picked)}개 문서 선택됨")
            
            # 컨텍스트 포맷팅
            lines = []
            for r in picked:
                meta = r.get("metadata") or {}
                page = meta.get("page")
                page_str = f"p.{page}" if page else "unknown"
                
                txt = (r.get("text") or "").replace("\n", " ").strip()
                if len(txt) > 600:
                    txt = txt[:600] + "..."
                
                lines.append(f"[{page_str}] {txt}")
            
            context = "\n\n".join(lines)
            print(f"📝 컨텍스트 길이: {len(context)} chars")
            
            return context
            
        except Exception as e:
            print(f"❌ Context 검색 중 에러: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _analyze(self) -> ComparisonState:
        """
        경쟁사 비교 분석 실행 (내부 메서드)
        
        Returns:
            ComparisonState: 분석 결과
        """
        print(f"\n{'='*50}")
        print(f"[ComparisonAgent] {self.company_name} 분석 시작")
        print(f"{'='*50}")
        
        # 기본 state 초기화 (에러 발생 시에도 필수 필드 보장)
        state: ComparisonState = {
            "company_name": self.company_name,
            "competitor_score": 0,
            "competitor_analysis_basis": ""
        }
        
        # 1. 인덱스 존재 여부 확인
        if not os.path.exists(self.chroma_dir):
            print("❌ Chroma 인덱스가 존재하지 않습니다")
            state["competitor_analysis_basis"] = (
                "❌ Index not found. Run build_index.py first."
            )
            return state
        
        # 2. Store 초기화 확인
        if not self.store:
            print("❌ HybridStore 초기화 실패")
            state["competitor_analysis_basis"] = (
                "❌ Failed to initialize HybridStore."
            )
            return state
        
        # 3. RAG 컨텍스트 구축
        print("\n[1/3] RAG 컨텍스트 검색 중...")
        try:
            context = self._build_context()
            
            if not context:
                print("❌ RAG 컨텍스트가 비어있습니다")
                state["competitor_analysis_basis"] = (
                    "❌ No context retrieved from RAG."
                )
                return state
                
        except Exception as e:
            print(f"❌ RAG 검색 실패: {e}")
            import traceback
            traceback.print_exc()
            state["competitor_analysis_basis"] = f"❌ RAG Error: {str(e)}"
            return state
        
        # 4. LLM 프롬프트 구성
        print("\n[2/3] LLM 프롬프트 생성 중...")
        
        sys_prompt = (
            "당신은 한국어로만 답변하는 스타트업 경쟁 분석 전문가입니다. "
            "정량적 근거를 기반으로 대상 기업의 경쟁 위치를 평가하고, "
            "0-100 정수 점수와 한국어 근거 요약을 제공합니다."
        )

        user_prompt = f"""대상 기업: {self.company_name}

아래 근거를 바탕으로 경쟁 위치를 분석하세요.

**출력 형식 (반드시 준수)**

1) 첫 줄: 점수만 정수로 표기
   예시: 75

2) 경쟁사 비교표 (마크다운 형식, 1개만 작성)
   - 반드시 대상 기업을 포함한 동종 경쟁사 비교
   - 주요 지표: MUV 추이, 자산, 부채, 자본 등
   - 깔끔한 1개의 표만 작성 (중복 금지)

3) 분석 요약 (3-5문장, 한국어)
   - 대상 기업의 강점
   - 대상 기업의 약점  
   - 경쟁사 대비 위치
   - 수치적 근거 포함

**근거 자료 (RAG)**
{context}

**주의사항**
- 표는 반드시 1개만 작성하고 중복하지 마세요
- 편향 없이 수치 중심으로 객관적으로 작성
- RAG 근거의 실제 수치를 정확히 인용
- 표가 깨지지 않도록 정렬에 주의
"""
        
        # 5. LLM 호출
        print("\n[3/3] LLM 분석 실행 중...")
        
        try:
            resp = self.llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = getattr(resp, "content", str(resp)) or ""
            
            if not content:
                print("⚠️ LLM 응답이 비어있습니다")
                state["competitor_analysis_basis"] = "❌ Empty LLM response"
                return state
            
            # 점수 및 요약 추출
            score, summary = _extract_score_and_summary(content)
            
            state["competitor_score"] = int(score)
            state["competitor_analysis_basis"] = summary.strip()
            
            print(f"\n✅ 분석 완료!")
            print(f"   점수: {score}/100")
            print(f"   요약 길이: {len(summary)} chars")
            
        except Exception as e:
            print(f"❌ LLM 호출 중 에러: {e}")
            import traceback
            traceback.print_exc()
            
            state["competitor_analysis_basis"] = f"❌ LLM Error: {str(e)}"
        
        return state

    def get_comparison_result(self) -> ComparisonState:
        """
        경쟁사 비교 분석 결과 반환
        
        main.py의 run_comparison_agent()에서 호출되는 메서드
        
        Returns:
            ComparisonState: 경쟁사 비교 분석 결과
        """
        try:
            result = self._analyze()
            return result
            
        except Exception as e:
            print(f"❌ ComparisonAgent 전체 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 에러 발생 시에도 올바른 구조의 state 반환
            return {
                "company_name": self.company_name,
                "competitor_score": 0,
                "competitor_analysis_basis": f"❌ Agent Error: {str(e)}"
            }


# ===== 하위 호환성 유지 (LangGraph 노드용) =====

def analyze_competitor(state: ComparisonState) -> ComparisonState:
    """
    Backward-compatible function for LangGraph nodes
    
    Args:
        state: ComparisonState with company_name
    
    Returns:
        ComparisonState: Updated state with analysis results
    """
    company_name = state.get("company_name", "Unknown")
    agent = ComparisonAgent(company_name=company_name)
    return agent.get_comparison_result()


# Alias
analyze_company_comparison = analyze_competitor


__all__ = ["ComparisonAgent", "analyze_competitor", "analyze_company_comparison"]