import os
from langchain_openai import ChatOpenAI
from state.final_state import FinalState


class ReportAgent:
    """보고서를 생성하는 Agent"""
    
    def __init__(self, final_state: FinalState):
        self.final_state = final_state
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.company_name = final_state["company_name"]
    
    def generate_report(self) -> str:
        """보고서 생성"""
        
        # 4번 섹션 생성 (기술력 비교)
        tech_section = self._generate_tech_section()
        
        # 전체 보고서 구성
        report = f"""# 기업 분석 보고서: {self.company_name}

---

## 1. 투자 개요

### 1.1 투자요약
[추후 MarketState 추가 시 작성 예정]

### 1.2 핵심 인사이트 및 레드플래그
[추후 MarketState 추가 시 작성 예정]

---

## 2. 시장성 분석

### 2.1 문제정의·고객세그먼트
[추후 MarketState 추가 시 작성 예정]

### 2.2 규제·거시 트렌드
[추후 MarketState 추가 시 작성 예정]

---

## 3. 경쟁업체 비교

### 3.1 경쟁지표(벤치마크, 대체재)
[추후 ComparisonState 추가 시 작성 예정]

### 3.2 포지셔닝 맵·진입장벽
[추후 ComparisonState 추가 시 작성 예정]

---

## 4. 기술력 비교

{tech_section}

---

## 5. 종합 요약

### 5.1 기업 투자 순위
[추후 전체 분석 완료 시 작성 예정]

### 5.2 기업 종합 평가 및 투자 전략
[추후 전체 분석 완료 시 작성 예정]

---

*보고서 생성 일시: {self._get_timestamp()}*
"""
        
        return report
    
    def _generate_tech_section(self) -> str:
        """4번 섹션 (기술력 비교) 생성"""
        
        tech_state = self.final_state["tech"]
        tech_evaluations = tech_state["tech_evaluations"]
        
        prompt = f"""당신은 스타트업 기술 분석 전문가입니다. 다음 정보를 바탕으로 기술력 비교 섹션을 작성해주세요.

=== 회사 정보 ===
회사명: {self.company_name}
현재 평가 대상: {tech_state.get('current_startup', 'N/A')}
평가 대상 스타트업들: {tech_state.get('startup_names', [])}

=== 기술 평가 결과 ===
기술 점수: {tech_evaluations['tech_score']}점
평가 근거: {tech_evaluations['score_basis']}

=== 추가 수집 데이터 ===
웹 크롤링 데이터: {tech_state.get('web_data', 'N/A')[:500]}...
검색된 문서 수: {len(tech_state.get('retrieved_docs', []))}개

---

다음 형식으로 작성해주세요:

### 4.1 제품 로드맵·사용자 여정
- 회사의 기술 발전 방향과 제품 로드맵을 분석
- 사용자 여정 관점에서의 기술 적용 사례
- 수집된 데이터를 바탕으로 구체적으로 작성

### 4.2 품질/안전
- 기술 점수 및 근거 요약
- 품질 관리 체계 및 안전성 평가
- 기술적 강점과 개선 필요 사항

**중요**: 
- 마크다운 형식으로 작성
- 각 섹션은 3-5문단으로 구성
- 구체적이고 전문적인 톤으로 작성
- 평가 근거를 명확히 제시
"""
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _get_timestamp(self):
        """현재 시각 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")