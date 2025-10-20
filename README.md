# AI Startup Investment Evaluation Agent

---

## Overview

“여행 산업의 AI 스타트업 **기술 경쟁력, 시장 기회, 경쟁사 대비 위치**를 바탕으로 **투자 판단 보고서 자동생성”**

---

## Why the Travel Domain?

> “왜 여행 도메인을 설정했는가?”

1. **복잡성과 실패율이 높은 도메인이기 때문**

   여행 스타트업은 **높은 진입률 대비 실패율이 가장 높은 산업 중 하나**입니다.

   특히 온라인 여행 분야는 다음과 같은 구조적 장벽이 존재합니다:

   - 복잡한 유통망 (GDS, 항공 예약망, 결제 시스템 등)
   - 뚜렷한 선두 기업 존재(마이리얼트립, 트립닷컴…) → 차별화 어려움
   - 고객 획득 비용이 매우 높아 성공 예측이 어려운 구조
     - 예: 익스피디아는 전체 비용의 약 50%를 광고에 사용→ 신규 스타트업은 고객 한 명을 확보하는 데 **수만~수십만 원 수준 소모**

   [[출처: TNMT by Lufthansa Innovation Hub](https://tnmt.com/startup-graveyard/)]

2. **서비스와 채널이 존재해 정량 지표 확보가 용이하기 때문**

   대부분 앱/웹 기반 서비스이기 때문에 아래와 같은 **정량 데이터 확보가 용이**합니다:

   - MUV / MAU
   - 월별 거래액 / 건수
   - 고객군 (성별, 연령별), 유입 채널

   → 또한, 웹 서치 후 추가 지표 연동도 가능해 **Agent가 자동 평가·랭킹 실험을 수행하기 최적화된 도메인**입니다.

---

## Architecture

<img width="456" height="610" alt="Image" src="https://github.com/user-attachments/assets/caf1978d-acdc-47ca-bcd4-cd27b78d9691" />

---

## Agents

| Agent                      | 역할                                 | 주요 입력                                  | 주요 출력                                                                                |
| -------------------------- | ------------------------------------ | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **TechSummaryAgent**       | 기술 문서 요약 및 **기술 점수 산출** | `기술요약_전체_기업_인터뷰.pdf`            | `company_name`<br>`category_scores`<br>`technology_score`<br>`technology_analysis_basis` |
| **MarketEvaluationAgent**  | **시장성 평가**                      | `시장분석_스타트업_시장전략_및_생태계.pdf` | `company_name`<br>`market_score`<br>`market_analysis_basis`                              |
| **CompanyComparisonAgent** | **경쟁사 비교**                      | `기업비교.pdf`                             | `company_name`<br>`competitor_score`<br>`competitor_analysis_basis`                      |

---

## Features

- **PDF 기반 RAG 자동화**
  - `기술요약_전체_기업_인터뷰.pdf`, `시장분석_스타트업_시장전략_및_생태계.pdf`, `기업비교.pdf`
  - pypdf → 청크 → 텍스트 **임베딩(`nomic-embed-text`)** → VectorDB
- WebSearch 자동화
  - Tavily를 이용한 Web Search 및 기업 상태 판단
- **투자 기준별 판단 분류 (3개 핵심 Agent)**
  - **기술력:** `TechAgent` → 핵심 기술, 차별성, 전달 가치
  - **시장성:** `MarketEvaluationAgent` → 성장성, 경쟁 강도, 수요 신호
  - **경쟁 위치:** `CompanyComparisonAgent` → 재무·소비자·투자 내역 기반 **동종사 비교**
- **투자 판단 & 보고서 생성 플로우 (수정 필요)**
  - `Judge_Agent`: 3개 Agent 결과를 종합하여 **보고서 생성(적격) / 생성 보류(비적격)** 를 결정
  - `Report_Agent`: **투자 적격일 경우** 최종 보고서 자동 생성

---

## Why We Predefined the Startup Candidates?

> “Agent가 직접 유망 스타트업을 찾게 하지 않고, 왜 우리가 사전 정의한 리스트로 시작했는가?”

- 여행 산업은 **‘관광 스타트업’이라는 개념 자체가 불명확**합니다.
  - 기술 기반 vs 서비스 기반 기업이 뒤섞여 있음
  - 정부, 지자체, 민간의 평가 기준도 제각각
  - 산업 분류 체계 자체가 **노후화되고 오프라인 중심**
- 이로 인해 RAG Agent가 후보를 탐색하기엔 **판단 기준이 정리되어 있지 않음**
  - 어떤 기업을 대상으로 분석해야 할지 정의가 어려움
  - 정량 비교를 수행할 수 있는 최소 요건도 갖춰지지 않음
    → 그래서 저희는 **실험의 안정성과 비교 가능성을 확보**하기 위해 **검증된 5개 스타트업 후보를 수작업으로 큐레이션**한 뒤 평가 Agent를 설계했습니다.
    [[출처: 이데일리 「관광 스타트업의 눈물」, 2024]](https://www.edaily.co.kr/News/Read?newsId=02076246642173184&mediaCodeNo=257#:~:text=%EC%9D%BC%EA%B4%80%EB%90%98%EC%A7%80%20%EC%95%8A%EC%9D%80%20%EA%B8%B0%EC%A4%80%EC%9C%BC%EB%A1%9C%20%EA%B8%B0%EC%97%85%EC%9D%84%20%EC%84%A0%EB%B3%84%ED%95%B4,%EC%A7%80%EC%9B%90%ED%95%98%EA%B3%A0%20%EC%9E%88%EB%8B%A4%EB%8A%94%20%EA%B2%83%EC%9D%B4%EB%8B%A4)

---

## Why Hybrid (Semantic + BM25)? — 의사결정 근거

1. **정량 근거 보존 필요**

   예를 들어, `기업비교.pdf`에는 재무/거래/MUV/투자 내역 등 **정확한 수치**가 다수 존재합니다.

   단일 임베딩 검색만으로는 특정 수치·항목 키워드(예: “MUV”, “자산”, “거래액”)의 **정확 매칭**이 어렵습니다.

2. **문맥 + 키워드 결합**

   Semantic은 **의미적 유사도**, BM25는 **키워드 정확 일치**에 강점이 있습니다.

   두 신호를 **가중 평균**으로 결합하면 “대상 기업 vs 동종 경쟁사” 비교 시 **수치 누락·왜곡을 줄이고 재현성**을 확보할 수 있습니다.

---

## Directory Structure

```
startsup-analysis-report/
│
├─ agents/
│   ├─ tech_summary_agent.py          # 기술 문서 요약 및 핵심 기술 도출 Agent
│   ├─ market_evaluation_agent.py     # 시장성 평가 Agent (시장 트렌드/수요 분석)
│   └─ company_comparison_agent.py    # 기업 간 비교 분석 Agent (재무·경영 지표 기반)
│
├─ rag/
│   ├─ tech/                          # 기술 관련 RAG 파이프라인
│   │   ├─ init.py
│   │   ├─ build_index.py             # PDF 로드 및 ChromaDB 인덱스 구축
│   │   ├─ description.txt            # 해당 인덱스 목적 및 설명 파일
│   │   └─ .chroma/                   # 기업비교용 ChromaDB 벡터 저장 폴더
│   │
│   ├─ market/                        # 시장 분석용 RAG 파이프라인
│   │   ├─ init.py
│   │   ├─ build_index.py             # PDF 로드 및 ChromaDB 인덱스 구축
│   │   ├─ description.txt            # 해당 인덱스 목적 및 설명 파일
│   │   └─ .chroma/                   # 기업비교용 ChromaDB 벡터 저장 폴더
│   │
│   ├─ company_comparison/            # 기업 비교용 RAG 파이프라인
│   │   ├─ init.py
│   │   ├─ build_index.py             # PDF 로드 및 ChromaDB 인덱스 구축
│   │   ├─ hybrid_store.py            # BM25 + Embedding 하이브리드 검색 구현
│   │   ├─ description.txt            # 해당 인덱스 목적 및 설명 파일
│   │   └─ .chroma/                   # 기업비교용 ChromaDB 벡터 저장 폴더
│   │
│   └─ init.py
│
├─ state/                             # LangGraph에서 각 Agent별 상태 관리 클래스
│   ├─ init.py
│   ├─ comparison_state.py             # 기업 비교 Agent의 상태 정의 (startup_name, score, etc.)
│   ├─ market_state.py                 # 시장 평가 Agent의 상태 정의
│   ├─ tech_state.py                  # 기술 요약 Agent의 상태 정의
│   ├─ judge_state.py                 # 종합 판단 및 평가 점수 통합용 State
│   ├─ workflow_state.py              # 전체 워크플로우 관리용 State
│   └─ final_state.py                 # 최종 통합 결과 (report generation 등)
│
├─ data/
│   ├─ 기업비교.pdf                   # 기업 비교용 PDF (재무/투자/소비자 데이터 포함)
│   ├─ 기술요약_전체_기업_인터뷰.pdf   # 기술 요약 PDF (각 회사의 중요 기술)
│   └─ 시장성분석_스타트업_시장전략_및_생태계.pdf # 시장 분석용 PDF (여행 산업 트렌드)
│
├─ llm/
│   ├─ judge_llm.py                   # 기업 보고서 생성 판단 LLM
│   └─ report_llm.py                  # 기업 보고서 생성 LLM
|
├─ reports/
│   ├─ report.pdf                     # 투자 판단 보고서(승인) PDF
│   ├─ report.md                      # 투자 판단 보고서(승인) MD
│   └─ reject.txt                     # 투자 판단 보고서(거절) TXT
│
├─ .env                               # openai api key, tavily api key 설정
├─ .gitignore                         # git ignore 대상 지정
├─ main.py                            # LangGraph 실행 진입점 (노드·엣지 정의 및 invoke 실행)
├─ README.md                          # 프로젝트 개요, 실행 방법, Agent 설명
└─ requirements.txt                   # 필수 라이브러리 정의
```

---

## Tech Stack

| Category         | Details                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------- |
| **Framework**    | LangGraph, LangChain, Python                                                                |
| **LLM**          | OpenAI `gpt-4o`                                                                             |
| **Retrieval**    | Hybrid Retrieval (Semantic+ BM25)                                                           |
| **Embedding**    | OllamaEmbeddings `(model="nomic-embed-text")`                                               |
| **Vector Store** | ChromaDB                                                                                    |
| **Data Source**  | `기술요약_전체_기업_인터뷰.pdf`, `시장분석_스타트업_시장전략_및_생태계.pdf`, `기업비교.pdf` |

---

## Contributors

- **권예지** – **기업비교(CompanyComparison) Agent**, README Documentation (Author)
- **이준희** – **기술요약(Tech) Agent**, README Graph Construction (Architect)
- **신호준** – **시장성평가(MarketEvaluation) Agent**, 공동 개발환경 구성 (GitHub / Requirements 관리)
- **이의진** – **전체 데이터 수집,** **투자판단(Judge) · 보고서생성(Report) LLM**

---
