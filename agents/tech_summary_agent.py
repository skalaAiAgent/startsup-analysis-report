# tech state # 기술 문서 요약 및 핵심 기술 도출을 수행하는 Agentimport os
from typing import TypedDict, List, Dict, Annotated
from operator import add
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
from dotenv import load_dotenv

load_dotenv()

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # 점수 다양성을 위해 temperature 증가
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 현재 설치된 임베딩 모델

# PDF 데이터 경로
PDF_DATA_PATH = Path(r"C:\workspace\demo-app\skala_gai\data")

class TechState(TypedDict):
    """에이전트의 상태를 정의하는 클래스"""
    startup_names: List[str]  # 평가할 스타트업 이름 리스트 (최대 5개)
    current_startup: str  # 현재 평가 중인 스타트업
    web_data: str  # 웹 크롤링 데이터
    retrieved_docs: List[Document]  # 검색된 문서들
    tech_evaluations: List[Dict]  # 기술 평가 결과들 - Annotated 제거하고 일반 List로 변경
    processing_index: int  # 현재 처리 중인 인덱스
    vectorstore_ready: bool  # VectorStore 준비 완료 여부
    #

class Tech_functions():
    def load_pdf_documents(pdf_dir: Path) -> List[Document]:
        """PDF 문서들을 로드하고 청킹"""
        all_documents = []
        
        # PDF 파일들 찾기
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"발견된 PDF 파일: {len(pdf_files)}개")
        
        for pdf_file in pdf_files:
            try:
                print(f"로딩 중: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # 메타데이터 추가
                for doc in documents:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["source_type"] = "pdf"
                
                all_documents.extend(documents)
            except Exception as e:
                print(f"PDF 로드 실패 ({pdf_file.name}): {e}")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        split_documents = text_splitter.split_documents(all_documents)
        print(f"총 {len(split_documents)}개의 청크 생성")
        
        return split_documents
    

    def crawl_startup_info(startup_name: str, max_results: int = 5) -> str:
        """
        스타트업 정보를 웹에서 크롤링 (Tavily)
        
        Args:
            startup_name: 스타트업 이름
            max_results: 수집할 최대 결과 수
        
        Returns:
            크롤링된 텍스트 데이터
        """
        print(f"\n{'='*60}")
        print(f"웹 검색 시작: {startup_name}")
        print(f"{'='*60}")
        
        # 1순위: Tavily 시도
        try:
            result = crawl_with_tavily(startup_name, max_results)
            if result and len(result) > 100:  # 유의미한 결과
                print(f"✓ Tavily 검색 성공\n")
                return result
            else:
                print(f"✗ Tavily 결과 부족, 대체 방법 시도...\n")
        except Exception as e:
            print(f"✗ Tavily 오류: {str(e)[:50]}")


    def crawl_with_tavily(startup_name: str, max_results: int) -> str:
        """
        Tavily API를 사용한 검색 (LLM 최적화)
        
        Args:
            startup_name: 스타트업 이름
            max_results: 수집할 최대 결과 수
        
        Returns:
            검색된 텍스트 데이터
        """
        from tavily import TavilyClient
        
        # API 키 확인
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")
        
        print(f"  [Tavily] 검색 중...")
        
        # 클라이언트 초기화
        client = TavilyClient(api_key=api_key)
        
        # 검색 쿼리 구성
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
                    search_depth="basic",  # "basic" 또는 "advanced"
                    max_results=max_results,
                    include_answer=True,  # AI 생성 요약 포함
                    include_raw_content=False,
                    include_domains=None,  # 모든 도메인 허용
                    days=365  # 최근 1년 데이터
                )
                
                # AI 생성 요약 (매우 중요!)
                if response.get('answer'):
                    collected_text.append(
                        f"[AI 요약] {response['answer']}"
                    )
                    print(f"      ✓ AI 요약 수집")
                
                # 검색 결과
                results = response.get('results', [])
                print(f"      ✓ {len(results)}개 출처 발견")
                
                for result in results:
                    title = result.get('title', '')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    score = result.get('score', 0)
                    
                    if content and len(content) > 50:
                        collected_text.append(
                            f"[출처: {url}]\n"
                            f"제목: {title}\n"
                            f"내용: {content}\n"
                            f"관련성: {score:.2f}"
                        )
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"      ✗ 쿼리 실패: {str(e)[:50]}")
                continue
        
        if not collected_text:
            raise ValueError("Tavily에서 유의미한 결과를 찾지 못했습니다.")
        
        result_text = "\n\n".join(collected_text)
        print(f"    총 {len(collected_text)}개 항목 수집")
        
        return result_text
    
    def initialize_vectorstore():
        """ChromaDB 벡터스토어 초기화 및 EnsembleRetriever 구성"""
        import os.path
        import shutil

        # ====== 설정 옵션 ======
        FORCE_REBUILD = False  # True로 설정하면 기존 DB 삭제하고 재생성
        # =======================

        CHROMA_PERSIST_DIR = "../rag/tech/chroma_db"
        CHROMA_COLLECTION_NAME = "startup_tech_db"

        # 강제 재생성 옵션이 활성화된 경우
        if FORCE_REBUILD and os.path.exists(CHROMA_PERSIST_DIR):
            print("⚠️ FORCE_REBUILD=True: 기존 VectorStore를 삭제합니다...")
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print("✓ 삭제 완료\n")

        # 이미 ChromaDB가 존재하는지 확인
        if os.path.exists(CHROMA_PERSIST_DIR) and os.path.isdir(CHROMA_PERSIST_DIR):
            print("=" * 60)
            print("📂 기존 VectorStore 발견!")
            print("=" * 60)
            print("저장된 임베딩 데이터를 로드합니다 (임베딩 생성 생략)...\n")
            
            # 기존 ChromaDB 로드 (PDF 로드 및 임베딩 생성 생략)
            vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            
            print(f"✓ VectorStore 로드 완료!")
            
            # PDF 문서도 로드 (BM25용으로 필요)
            print("\nPDF 문서 로드 중 (BM25 인덱스용)...")
            pdf_documents = load_pdf_documents(PDF_DATA_PATH)
            
        else:
            print("=" * 60)
            print("🆕 기존 VectorStore 없음 - 새로 생성")
            print("=" * 60)
            print("임베딩 생성 중 (처음 실행 시 시간 소요)...\n")
            
            # PDF 문서 로드
            print("PDF 문서 로드 중...")
            pdf_documents = load_pdf_documents(PDF_DATA_PATH)
            
            # ChromaDB 벡터스토어 생성
            print("\nVectorStore 생성 중 (임베딩 생성 - 수 분 소요 가능)...")
            vectorstore = Chroma.from_documents(
                documents=pdf_documents,
                embedding=embeddings,
                collection_name=CHROMA_COLLECTION_NAME,
                persist_directory=CHROMA_PERSIST_DIR
            )
            
            print("✓ VectorStore 생성 완료!")

        # ========== EnsembleRetriever 구성 ==========
        print("\n" + "=" * 60)
        print("🔧 EnsembleRetriever 구성 중...")
        print("=" * 60)

        # 1. BM25Retriever 생성 (키워드 기반)
        bm25_retriever = BM25Retriever.from_documents(pdf_documents)
        bm25_retriever.k = 5  # 상위 5개 문서 반환

        print(f"✓ BM25Retriever 생성 완료 (k={bm25_retriever.k})")

        # 2. Semantic Retriever 생성 (벡터 기반)
        semantic_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        print(f"✓ SemanticRetriever 생성 완료 (k=5)")

        # 3. EnsembleRetriever로 결합
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]  # 동일한 가중치
        )

        print(f"✓ EnsembleRetriever 생성 완료 (weights=[0.5, 0.5])")

        print(f"\n{'='*60}")
        print(f"✅ 초기화 완료")
        print(f"{'='*60}")
        print(f"PDF 문서 수: {len(pdf_documents)}개")
        print(f"컬렉션 이름: {CHROMA_COLLECTION_NAME}")
        print(f"저장 위치: {CHROMA_PERSIST_DIR}")
        print(f"Retriever 구성: BM25 (50%) + Semantic (50%)")
        print(f"{'='*60}\n")

    def select_next_startup(state: TechState) -> TechState:
        """다음 평가할 스타트업 선택"""
        idx = state.get("processing_index", 0)
        
        if idx < len(state["startup_names"]):
            state["current_startup"] = state["startup_names"][idx]
            # processing_index는 여기서 업데이트하지 않음 (evaluate에서 업데이트)
            print(f"\n{'='*60}")
            print(f"[{idx+1}/{len(state['startup_names'])}] {state['current_startup']} 평가 시작")
            print(f"{'='*60}")
        
        return state


    def crawl_web_data(state: TechState) -> TechState:
        """웹에서 스타트업 정보 크롤링"""
        startup_name = state["current_startup"]
        
        # 웹 크롤링
        web_data = crawl_startup_info(startup_name, max_results=5)
        state["web_data"] = web_data
        
        return state


    def retrieve_tech_info(state: TechState) -> TechState:
        """PDF 문서에서 관련 기술 정보 검색"""
        startup_name = state["current_startup"]
        
        # 검색 쿼리 구성 (스타트업 특성 고려)
        query = f"{startup_name} AI 기술 혁신 스타트업 투자 평가 경쟁력"
        
        print(f"\nPDF 문서에서 관련 정보 검색 중...")
        # EnsembleRetriever로 검색 수행
        retrieved_docs = ensemble_retriever.get_relevant_documents(query)
        
        state["retrieved_docs"] = retrieved_docs
        print(f"검색 완료: {len(retrieved_docs)}개 문서 검색됨")
        
        return state


    def evaluate_technology(state: TechState) -> TechState:
        """웹 데이터와 PDF 정보를 바탕으로 기술력 평가"""
        startup_name = state["current_startup"]
        web_data = state.get("web_data", "정보 없음")
        docs = state["retrieved_docs"]
        current_index = state.get("processing_index", 0)
        
        # 이미 평가된 기업들의 점수 확인
        existing_evaluations = state.get("tech_evaluations", [])
        existing_scores = [e['기술_점수'] for e in existing_evaluations if isinstance(e, dict)]
        
        # PDF 문서를 컨텍스트로 결합
        pdf_context = "\n\n".join([doc.page_content for doc in docs[:3]])  # 상위 3개
        
        print(f"\nGPT-4o-mini를 사용하여 기술력 평가 중...")
        print(f"  현재까지 평가 완료: {len(existing_evaluations)}개")
        
        # 기존 점수 정보를 프롬프트에 강화
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
        
        # 평가 프롬프트 (단계별 세부 평가 요구)
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
        
        # LLM 호출
        chain = eval_prompt | llm
        response = chain.invoke({
            "startup_name": startup_name,
            "web_data": web_data[:2000],  # 토큰 제한 고려
            "pdf_context": pdf_context[:3000],
            "existing_scores_constraint": existing_scores_constraint
        })
        
        # JSON 파싱
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            evaluation = json.loads(content.strip())
            
            # 항목별 점수가 있으면 총점 검증
            if "항목별_점수" in evaluation:
                item_scores = evaluation["항목별_점수"]
                calculated_total = sum(item_scores.values())
                reported_total = evaluation.get("기술_점수", calculated_total)
                
                # 총점이 맞지 않으면 재계산
                if abs(calculated_total - reported_total) > 1:
                    print(f"  ⚠️ 점수 불일치 감지 (보고: {reported_total}, 계산: {calculated_total}) - 재계산된 값 사용")
                    evaluation["기술_점수"] = calculated_total
            
            print(f"평가 완료: {evaluation['기술_점수']}점")
            
            # 항목별 점수 출력
            if "항목별_점수" in evaluation:
                scores_breakdown = ", ".join([f"{k}={v}" for k, v in evaluation["항목별_점수"].items()])
                print(f"  세부: {scores_breakdown}")
            
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")
            print(f"응답 내용: {response.content[:300]}")
            evaluation = {
                "startup_name": startup_name,
                "기술_점수": 50,  # 기본값
                "기술_분석_근거": f"평가 실패: {str(e)}"
            }
        
        # ★ 중요: 기존 리스트에 append (덮어쓰기 아님!)
        current_evaluations = state.get("tech_evaluations", [])
        current_evaluations.append(evaluation)
        state["tech_evaluations"] = current_evaluations
        
        # processing_index 증가 (다음 기업으로)
        state["processing_index"] = current_index + 1
        
        print(f"진행 상황: {state['processing_index']}/{len(state['startup_names'])} 완료")
        print(f"  누적 평가 결과: {len(state['tech_evaluations'])}개\n")
        
        return state


    def check_completion(state: TechState) -> str:
        """모든 스타트업 평가 완료 여부 확인"""
        idx = state.get("processing_index", 0)
        total = len(state.get("startup_names", []))
        
        print(f"[check_completion] 현재 인덱스: {idx}, 전체: {total}")
        
        if idx < total:
            return "continue"  # 다음 스타트업 평가
        else:
            return "end"  # 모든 평가 완료

# StateGraph 생성
tech_workflow = StateGraph(TechState)
    
# 노드 추가
tech_workflow.add_node("select_startup", select_next_startup)
tech_workflow.add_node("crawl_web", crawl_web_data)
tech_workflow.add_node("retrieve_info", retrieve_tech_info)
tech_workflow.add_node("evaluate", evaluate_technology)

# 엣지 설정
tech_workflow.set_entry_point("select_startup")
tech_workflow.add_edge("select_startup", "crawl_web")
tech_workflow.add_edge("crawl_web", "retrieve_info")
tech_workflow.add_edge("retrieve_info", "evaluate")

# 조건부 엣지 (평가 완료 후 다음 스타트업으로 이동 또는 종료)
tech_workflow.add_conditional_edges(
    "evaluate",
    check_completion,
    {
        "continue": "select_startup",
        "end": END
    }
)

# 그래프 컴파일
app = tech_workflow.compile()

print("\n워크플로우 구성 완료!")
print("순서: select_startup -> crawl_web -> retrieve_info -> evaluate -> [반복 or 종료]")

# 평가할 스타트업 리스트
startups_to_evaluate = [
    "리브 애니웨어",
    "어딩",
    "트립비토즈",
    "트립소다",
    "하이어플레이스"
]

# 초기 상태 설정 (명확한 초기화)
initial_state = {
    "startup_names": startups_to_evaluate,
    "current_startup": "",
    "web_data": "",
    "retrieved_docs": [],
    "tech_evaluations": [],  # 빈 리스트로 명확히 초기화
    "processing_index": 0,
    "vectorstore_ready": True
}

# 에이전트 실행
print("\n" + "=" * 60)
print("AI 스타트업 기술 평가 에이전트 시작")
print(f"평가 대상: {len(startups_to_evaluate)}개 기업")
print("=" * 60)

# 그래프 재컴파일 (이전 상태 초기화)
app = workflow.compile()

result = app.invoke(initial_state)

print("\n" + "=" * 60)
print("전체 평가 완료")
print(f"최종 평가 결과 수: {len(result['tech_evaluations'])}개")
print("=" * 60)


# 평가 결과 출력
evaluations = result["tech_evaluations"]

print(f"\n\n{'#'*60}")
print(f"총 {len(evaluations)}개 스타트업 평가 완료")
print(f"{'#'*60}\n")

for i, eval_data in enumerate(evaluations, 1):
    print(f"\n{'='*60}")
    print(f"[{i}] {eval_data['startup_name']}")
    print(f"{'='*60}")
    print(f"\n기술 점수: {eval_data['기술_점수']}/100점")
    print(f"\n분석 근거:")
    print(f"{eval_data['기술_분석_근거']}")
    print()