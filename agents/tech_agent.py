"""
AI 스타트업 기술 평가 에이전트
Langgraph를 사용하여 웹 크롤링 + PDF 분석을 통한 기술력 평가
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, List, Dict
import json
import time

# 프로젝트 루트를 Python 경로에 추가
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


class TechState(TypedDict):
    """에이전트의 상태를 정의하는 클래스"""
    startup_names: List[str]
    current_startup: str
    web_data: str
    retrieved_docs: List[Document]
    tech_evaluations: List[Dict]
    processing_index: int
    vectorstore_ready: bool


class TechAgent:
    """
    AI 스타트업 기술 평가 에이전트

    사용법:
        agent = TechAgent(startups_to_evaluate="어딩")
        result = agent.get_tech_result()
    """

    def __init__(self, startups_to_evaluate: str | List[str], pdf_data_path: str = None):
        """
        Args:
            startups_to_evaluate: 평가할 스타트업 이름 (문자열 또는 리스트)
            pdf_data_path: PDF 데이터 경로 (기본값: 프로젝트 루트의 data 폴더)
        """
        # 스타트업 리스트 설정
        if isinstance(startups_to_evaluate, str):
            self.startup_names = [startups_to_evaluate]
        else:
            self.startup_names = startups_to_evaluate

        # PDF 데이터 경로 설정
        if pdf_data_path is None:
            # 프로젝트 루트의 data 폴더 사용
            self.pdf_data_path = Path(ROOT_DIR).parent / "data"
        else:
            self.pdf_data_path = Path(pdf_data_path)

        # 모델 초기화
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # ChromaDB 경로
        self.chroma_persist_dir = os.path.join(ROOT_DIR, "rag", "tech", "chroma_db")
        self.chroma_collection_name = "startup_tech_db"

        # VectorStore 및 Retriever 초기화
        self.vectorstore = None
        self.ensemble_retriever = None
        self.pdf_documents = None

        # Workflow 초기화
        self.app = None

        print(f"\n{'='*60}")
        print(f"TechAgent 초기화")
        print(f"{'='*60}")
        print(f"평가 대상: {', '.join(self.startup_names)}")
        print(f"PDF 경로: {self.pdf_data_path}")
        print(f"{'='*60}\n")

    def _load_pdf_documents(self) -> List[Document]:
        """PDF 문서들을 로드하고 청킹"""
        all_documents = []

        pdf_files = list(self.pdf_data_path.glob("*.pdf"))
        print(f"발견된 PDF 파일: {len(pdf_files)}개")

        for pdf_file in pdf_files:
            try:
                print(f"  로딩 중: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["source_type"] = "pdf"

                all_documents.extend(documents)
            except Exception as e:
                print(f"  PDF 로드 실패 ({pdf_file.name}): {e}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        split_documents = text_splitter.split_documents(all_documents)
        print(f"총 {len(split_documents)}개의 청크 생성\n")

        return split_documents

    def _initialize_vectorstore(self):
        """VectorStore 및 EnsembleRetriever 초기화"""
        print(f"{'='*60}")
        print(f"VectorStore 초기화")
        print(f"{'='*60}\n")

        # 기존 ChromaDB 확인
        if os.path.exists(self.chroma_persist_dir) and os.path.isdir(self.chroma_persist_dir):
            print("📂 기존 VectorStore 발견 - 로드 중...")
            self.vectorstore = Chroma(
                collection_name=self.chroma_collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_dir
            )
            print("✓ VectorStore 로드 완료\n")

            print("PDF 문서 로드 중 (BM25 인덱스용)...")
            self.pdf_documents = self._load_pdf_documents()
        else:
            print("🆕 기존 VectorStore 없음 - 새로 생성")
            print("PDF 문서 로드 중...")
            self.pdf_documents = self._load_pdf_documents()

            print("VectorStore 생성 중 (임베딩 생성 - 수 분 소요 가능)...")
            self.vectorstore = Chroma.from_documents(
                documents=self.pdf_documents,
                embedding=self.embeddings,
                collection_name=self.chroma_collection_name,
                persist_directory=self.chroma_persist_dir
            )
            print("✓ VectorStore 생성 완료\n")

        # EnsembleRetriever 구성
        print(f"{'='*60}")
        print(f"EnsembleRetriever 구성 중...")
        print(f"{'='*60}\n")

        bm25_retriever = BM25Retriever.from_documents(self.pdf_documents)
        bm25_retriever.k = 5

        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]
        )

        print("✓ BM25Retriever 생성 완료 (k=5)")
        print("✓ SemanticRetriever 생성 완료 (k=5)")
        print("✓ EnsembleRetriever 생성 완료 (weights=[0.5, 0.5])\n")

    def _crawl_with_tavily(self, startup_name: str, max_results: int = 5) -> str:
        """Tavily API를 사용한 웹 검색"""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")

        print(f"  [Tavily] 검색 중...")
        client = TavilyClient(api_key=api_key)

        queries = [
            f"{startup_name} 스타트업 기술 혁신",
            f"{startup_name} AI 투자 비즈니스"
        ]

        collected_text = []

        for idx, query in enumerate(queries, 1):
            try:
                print(f"    쿼리 {idx}: {query}")

                response = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=max_results,
                    include_answer=True,
                    include_raw_content=False,
                    include_domains=None,
                    days=365
                )

                if response.get('answer'):
                    collected_text.append(f"[AI 요약] {response['answer']}")
                    print(f"      ✓ AI 요약 수집")

                results = response.get('results', [])
                print(f"      ✓ {len(results)}개 출처 발견")

                for result in results:
                    title = result.get('title', '')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    score = result.get('score', 0)

                    if content and len(content) > 50:
                        collected_text.append(
                            f"[출처: {url}]\n제목: {title}\n내용: {content}\n관련성: {score:.2f}"
                        )

                time.sleep(0.3)
            except Exception as e:
                print(f"      ✗ 쿼리 실패: {str(e)[:50]}")
                continue

        if not collected_text:
            raise ValueError("Tavily에서 유의미한 결과를 찾지 못했습니다.")

        result_text = "\n\n".join(collected_text)
        print(f"    총 {len(collected_text)}개 항목 수집")
        return result_text

    def _crawl_startup_info(self, startup_name: str, max_results: int = 5) -> str:
        """스타트업 정보를 웹에서 크롤링"""
        print(f"\n{'='*60}")
        print(f"웹 검색 시작: {startup_name}")
        print(f"{'='*60}")

        try:
            result = self._crawl_with_tavily(startup_name, max_results)
            if result and len(result) > 100:
                print(f"✓ Tavily 검색 성공\n")
                return result
        except Exception as e:
            print(f"✗ Tavily 오류: {str(e)[:50]}")
            return f"웹 검색 실패: {str(e)}"

    def _build_workflow(self):
        """LangGraph 워크플로우 구성"""

        def select_next_startup(state: TechState) -> TechState:
            """다음 평가할 스타트업 선택"""
            idx = state.get("processing_index", 0)

            if idx < len(state["startup_names"]):
                state["current_startup"] = state["startup_names"][idx]
                print(f"\n{'='*60}")
                print(f"[{idx+1}/{len(state['startup_names'])}] {state['current_startup']} 평가 시작")
                print(f"{'='*60}")

            return state

        def crawl_web_data(state: TechState) -> TechState:
            """웹에서 스타트업 정보 크롤링"""
            startup_name = state["current_startup"]
            web_data = self._crawl_startup_info(startup_name, max_results=5)
            state["web_data"] = web_data
            return state

        def retrieve_tech_info(state: TechState) -> TechState:
            """PDF 문서에서 관련 기술 정보 검색"""
            startup_name = state["current_startup"]
            query = f"{startup_name} AI 기술 혁신 스타트업 투자 평가 경쟁력"

            print(f"\nPDF 문서에서 관련 정보 검색 중...")
            retrieved_docs = self.ensemble_retriever.get_relevant_documents(query)

            state["retrieved_docs"] = retrieved_docs
            print(f"검색 완료: {len(retrieved_docs)}개 문서 검색됨")

            return state

        def evaluate_technology(state: TechState) -> TechState:
            """웹 데이터와 PDF 정보를 바탕으로 기술력 평가"""
            startup_name = state["current_startup"]
            web_data = state.get("web_data", "정보 없음")
            docs = state["retrieved_docs"]
            current_index = state.get("processing_index", 0)

            existing_evaluations = state.get("tech_evaluations", [])
            existing_scores = [e['기술_점수'] for e in existing_evaluations if isinstance(e, dict)]

            pdf_context = "\n\n".join([doc.page_content for doc in docs[:3]])

            print(f"\nGPT-4o-mini를 사용하여 기술력 평가 중...")
            print(f"  현재까지 평가 완료: {len(existing_evaluations)}개")

            existing_scores_constraint = ""
            if existing_scores:
                scores_str = ", ".join(str(s) for s in existing_scores)
                existing_scores_constraint = f"""
### ⚠️ 중요한 제약 조건 ⚠️
이미 평가한 기업들의 점수: [{scores_str}]

**필수**: 새로운 기술_점수는 위 점수들과 **최소 5점 이상 차이**가 나야 합니다.
- 이미 사용된 점수: {existing_scores}
- 사용 금지 범위: {', '.join(f'{s}±4점' for s in existing_scores)}
- 각 기업의 실제 강점과 약점을 반영하여 차별화된 점수를 부여하세요.
"""

            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 AI 스타트업 투자 전문가입니다.
주어진 웹 정보와 업계 트렌드 문서를 바탕으로 스타트업의 기술력을 객관적으로 평가하세요.

## 평가 기준 (단계별 평가):

**1단계: 각 항목별 점수 산정**
- 기술의 혁신성 (0-30점): AI 기술의 독창성, 차별화된 접근 방식
- 기술의 완성도 (0-30점): 제품/서비스의 완성도, 실제 적용 사례
- 시장 경쟁력 (0-20점): 경쟁사 대비 우위, 시장 포지셔닝
- 특허/지식재산권 (0-10점): 특허, 논문, 기술 자산
- 기술 확장 가능성 (0-10점): 스케일업 가능성, 다른 분야 적용

**2단계: 총점 계산**
위 5개 항목의 점수를 합산하여 최종 기술_점수를 도출하세요.

**중요**:
- 기업마다 명확히 차별화된 점수를 부여하세요
- 모든 기업에게 비슷한 점수를 주지 마세요
- 각 기업의 실제 강점과 약점을 정확히 반영하세요"""),
                ("user", """스타트업 이름: {startup_name}

=== 웹에서 수집한 정보 ===
{web_data}

=== 업계 트렌드 및 참고 문서 ===
{pdf_context}

{existing_scores_constraint}

위 정보를 바탕으로 **단계별로 평가**하고 다음 JSON 형식으로 결과를 작성하세요:

{{
    "startup_name": "스타트업 이름",
    "항목별_점수": {{
        "혁신성": 점수 (0-30),
        "완성도": 점수 (0-30),
        "경쟁력": 점수 (0-20),
        "특허": 점수 (0-10),
        "확장성": 점수 (0-10)
    }},
    "기술_점수": 총점 (0-100, 정수),
    "기술_분석_근거": "각 항목별 점수 산정 이유를 구체적으로 설명. 혁신성, 완성도, 경쟁력, 특허, 확장성 각각에 대해 웹 정보를 인용하여 상세히 분석"
}}

**필수**: 기술_점수는 항목별_점수의 합과 일치해야 합니다.""")
            ])

            chain = eval_prompt | self.llm
            response = chain.invoke({
                "startup_name": startup_name,
                "web_data": web_data[:2000],
                "pdf_context": pdf_context[:3000],
                "existing_scores_constraint": existing_scores_constraint
            })

            try:
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                evaluation = json.loads(content.strip())

                if "항목별_점수" in evaluation:
                    item_scores = evaluation["항목별_점수"]
                    calculated_total = sum(item_scores.values())
                    reported_total = evaluation.get("기술_점수", calculated_total)

                    if abs(calculated_total - reported_total) > 1:
                        print(f"  ⚠️ 점수 불일치 감지 (보고: {reported_total}, 계산: {calculated_total}) - 재계산된 값 사용")
                        evaluation["기술_점수"] = calculated_total

                print(f"평가 완료: {evaluation['기술_점수']}점")

                if "항목별_점수" in evaluation:
                    scores_breakdown = ", ".join([f"{k}={v}" for k, v in evaluation["항목별_점수"].items()])
                    print(f"  세부: {scores_breakdown}")

            except Exception as e:
                print(f"JSON 파싱 실패: {e}")
                evaluation = {
                    "startup_name": startup_name,
                    "기술_점수": 50,
                    "기술_분석_근거": f"평가 실패: {str(e)}"
                }

            current_evaluations = state.get("tech_evaluations", [])
            current_evaluations.append(evaluation)
            state["tech_evaluations"] = current_evaluations
            state["processing_index"] = current_index + 1

            print(f"진행 상황: {state['processing_index']}/{len(state['startup_names'])} 완료")
            print(f"  누적 평가 결과: {len(state['tech_evaluations'])}개\n")

            return state

        def check_completion(state: TechState) -> str:
            """모든 스타트업 평가 완료 여부 확인"""
            idx = state.get("processing_index", 0)
            total = len(state.get("startup_names", []))

            if idx < total:
                return "continue"
            else:
                return "end"

        # StateGraph 생성
        workflow = StateGraph(TechState)

        # 노드 추가
        workflow.add_node("select_startup", select_next_startup)
        workflow.add_node("crawl_web", crawl_web_data)
        workflow.add_node("retrieve_info", retrieve_tech_info)
        workflow.add_node("evaluate", evaluate_technology)

        # 엣지 설정
        workflow.set_entry_point("select_startup")
        workflow.add_edge("select_startup", "crawl_web")
        workflow.add_edge("crawl_web", "retrieve_info")
        workflow.add_edge("retrieve_info", "evaluate")

        # 조건부 엣지
        workflow.add_conditional_edges(
            "evaluate",
            check_completion,
            {
                "continue": "select_startup",
                "end": END
            }
        )

        # 그래프 컴파일
        self.app = workflow.compile()

        print("\n워크플로우 구성 완료!")
        print("순서: select_startup -> crawl_web -> retrieve_info -> evaluate -> [반복 or 종료]\n")

    def get_tech_result(self) -> Dict:
        """
        기술 평가 실행 및 결과 반환

        Returns:
            평가 결과 딕셔너리
            {
                "tech_evaluations": [...],  # 각 스타트업별 평가 결과
                "summary": {...}  # 요약 통계
            }
        """
        # VectorStore 초기화 (아직 안 되어 있으면)
        if self.vectorstore is None:
            self._initialize_vectorstore()

        # Workflow 구성 (아직 안 되어 있으면)
        if self.app is None:
            self._build_workflow()

        # 초기 상태 설정
        initial_state = {
            "startup_names": self.startup_names,
            "current_startup": "",
            "web_data": "",
            "retrieved_docs": [],
            "tech_evaluations": [],
            "processing_index": 0,
            "vectorstore_ready": True
        }

        # 에이전트 실행
        print(f"\n{'='*60}")
        print(f"AI 스타트업 기술 평가 에이전트 시작")
        print(f"평가 대상: {len(self.startup_names)}개 기업")
        print(f"{'='*60}")

        result = self.app.invoke(initial_state)

        print(f"\n{'='*60}")
        print(f"전체 평가 완료")
        print(f"최종 평가 결과 수: {len(result['tech_evaluations'])}개")
        print(f"{'='*60}\n")


        return result["tech_evaluations"].pop(), 
