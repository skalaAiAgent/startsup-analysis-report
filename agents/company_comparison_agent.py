from __future__ import annotations

import os
import sys

# 프로젝트 루트를 Python 경로에 추가 (import 전에 실행)
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import re
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from rag.company_comparison.hybrid_store import HybridStore


# LangGraph State 정의
class AgentState(TypedDict, total=False):
    """Multi-Agent 통합 State"""
    # 공통
    startup_name: str
    
    # 기업비교 Agent
    competitor_score: int
    competitor_analysis_basis: str


def _extract_score_and_summary(text: str) -> tuple[int, str]:
    """
    GPT 응답에서 점수와 요약을 추출합니다.
    
    Args:
        text: GPT 응답 텍스트
        
    Returns:
        (점수, 요약) 튜플
    """
    # 점수 패턴: "45점", "점수: 45", "**45점**", "평가: 45" 등
    score_patterns = [
        r'(?:점수[:\s]*)?[\*]*(\d{1,3})[\*]*점',  # "45점", "**45점**"
        r'(?:평가|점수)[:\s]+(\d{1,3})',           # "평가: 45", "점수: 45"
        r'\b(100|\d{1,2})\s*점',                   # "45 점"
        r'\b(100|\d{1,2})/100',                    # "45/100"
    ]
    
    score = 50  # 기본값
    
    for pattern in score_patterns:
        m = re.search(pattern, text)
        if m:
            score = int(m.group(1))
            score = max(0, min(100, score))
            break
    
    # 요약 추출: 전체 응답을 그대로 사용 (표 포함)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    
    # 점수만 있는 첫 줄 제거
    if lines and re.match(r'^[\*\s]*\d{1,3}[\*\s]*점?[\*\s]*$', lines[0]):
        lines = lines[1:]
    
    # 전체 내용 유지 (표 + 요약)
    return score, "\n".join(lines)


def analyze_competitor(state: AgentState) -> AgentState:
    """
    기업비교 Agent - LangGraph 노드 함수
    
    5개 기업(트립소다, 트립비토즈, 하이어플레이스, 리브애니웨어, 어딩)을
    재무/투자/소비자/MUV 지표 기반으로 표 형식으로 비교 분석합니다.
    
    Args:
        state: LangGraph State (startup_name 포함)
        
    Returns:
        업데이트된 State (competitor_score, competitor_analysis_basis 추가)
    """
    # 환경변수 로드
    load_dotenv()
    
    startup = state.get("startup_name") or "샘플기업"
    print(f"\n{'='*70}")
    print(f"🔍 [기업비교 Agent] 분석 시작: {startup}")
    print(f"{'='*70}")
    
    # 비교 대상 기업 리스트
    companies = ["트립소다", "트립비토즈", "하이어플레이스", "리브애니웨어", "어딩"]
    
    # HybridStore 초기화
    agent_dir = os.path.join(os.path.dirname(__file__), os.pardir, "rag", "company_comparison")
    chroma_dir = os.path.join(agent_dir, ".chroma")
    
    if not os.path.exists(chroma_dir):
        print(f"❌ 인덱스가 없습니다. 먼저 build_index.py를 실행하세요: {chroma_dir}")
        state["competitor_score"] = 0
        state["competitor_analysis_basis"] = "인덱스를 찾을 수 없습니다."
        return state
    
    print(f"📂 인덱스 로드 중...")
    store = HybridStore(chroma_path=chroma_dir, collection="company_comparison")
    
    # 검색 쿼리 생성 (모든 기업 + 주요 지표)
    query = f"{' '.join(companies)} 재무제표 매출 자산 부채 투자유치 소비자 성별 거래 MUV"
    print(f"🔎 검색 쿼리: {query}")
    
    # 하이브리드 검색
    results = store.search(query=query, k_bm25=15, k_vec=15, rrf_k=60, lambda_vec=0.3)
    
    # 상위 결과 필터링 (다양한 페이지 선호)
    seen_pages = set()
    picked = []
    
    for r in results:
        m = r.get("metadata", {}) or {}
        page = m.get("page")
        
        # 페이지 중복 최소화하되 충분한 데이터 확보
        if page in seen_pages and len(picked) >= 10:
            continue
        
        seen_pages.add(page)
        picked.append(r)
        
        if len(picked) >= 15:
            break
    
    # Context 구성
    context_lines = []
    for r in picked[:15]:
        meta = r.get("metadata", {}) or {}
        page = meta.get("page")
        page_str = f" (p.{page})" if page else ""
        
        txt = (r.get("text") or "").replace("\n", " ").strip()
        # 표 데이터는 길어도 유지
        if len(txt) > 500:
            txt = txt[:500] + "..."
        
        context_lines.append(f"[{page_str}] {txt}")
    
    context = "\n\n".join(context_lines)
    print(f"📋 검색된 근거 수: {len(picked)}")
    
    # GPT 호출 - 구조화된 비교 분석
    client = OpenAI()
    
    sys_prompt = """당신은 스타트업 경쟁 분석 전문가입니다. 
정량적 데이터를 바탕으로 5개 기업을 객관적으로 비교하고, 
명확한 순위와 근거를 제시합니다.

**중요:** 분석 대상 기업에게 유리한 편향 없이 객관적으로 평가하세요."""
    
    user_prompt = f"""# 기업 비교 분석 요청

**5개 비교 대상 기업:** 트립소다, 트립비토즈, 하이어플레이스, 리브애니웨어, 어딩
**이 중 분석 요청 기업:** {startup}

## 분석 지표
1. **재무 건전성** (자산, 부채, 자본, 매출 증가율)
2. **투자 유치** (투자 유치 금액, 투자 라운드)
3. **시장 성과** (MUV, 거래액, 소비자 수)
4. **소비자 특성** (성별 분포, 연령대)

## 출력 형식

### 1단계: 점수 (첫 줄)
- **반드시 5개 기업을 실제 데이터 기반으로 순위를 매긴 후** 점수 산정
- 1위: 100점, 2위: 75점, 3위: 50점, 4위: 25점, 5위: 0점
- {startup}가 1위가 아닐 수 있으며, 실제 데이터에 따라 순위가 결정됨
- 점수만 숫자로 명시 (예: "75")

### 2단계: 비교표 (마크다운 테이블)
**5개 기업 모두 포함하여 작성**
| 기업명 | 재무건전성 | 투자유치 | 시장성과 | 종합순위 |
|--------|-----------|---------|---------|---------|
| 트립소다 | ... | ... | ... | X위 |
| 트립비토즈 | ... | ... | ... | X위 |
| 하이어플레이스 | ... | ... | ... | X위 |
| 리브애니웨어 | ... | ... | ... | X위 |
| 어딩 | ... | ... | ... | X위 |

### 3단계: 요약 (3~5줄)
- **{startup}의 강점:**
- **{startup}의 약점:**
- **경쟁사 대비 포지션:**
- **순위 근거:** (왜 이 순위인지 정량 데이터로 설명)

## 근거 자료
{context}

**경고:** 
- {startup}를 무조건 1위로 만들지 마세요.
- 실제 데이터(자산, 매출, 투자액 등)를 비교하여 객관적으로 순위를 매기세요.
- 근거 자료에 있는 실제 수치를 활용하세요.
- 표에 5개 기업을 모두 포함하세요."""
    
    print("🤖 GPT 분석 중...")
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    
    content = completion.choices[0].message.content or ""
    
    # 점수 및 요약 파싱
    score, summary = _extract_score_and_summary(content)
    
    # State 업데이트
    state["competitor_score"] = int(score)
    state["competitor_analysis_basis"] = summary.strip()
    
    print(f"✅ 분석 완료 - 점수: {score}점")
    
    return state


# 하위 호환성을 위한 별칭
analyze_company_comparison = analyze_competitor


if __name__ == "__main__":
    # 단독 테스트 실행
    from typing import cast
    
    init_state: AgentState = {
        "startup_name": os.environ.get("STARTUP_NAME", "어딩")
    }
    
    result = analyze_competitor(init_state)
    
    print("\n" + "="*70)
    print("📊 기업비교 분석 결과")
    print("="*70)
    print(f"\n🎯 경쟁사 점수: {result.get('competitor_score')}점\n")
    print("📋 분석 내용:")
    print("-"*70)
    print(result.get("competitor_analysis_basis", "").strip())
    print("="*70)