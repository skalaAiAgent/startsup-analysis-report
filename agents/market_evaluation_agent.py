# agents/market_evaluation_agent.py
"""
Market Evaluation Agent (LangGraph, class-based API)

외부 사용 예:
>>> from agents.market_evaluation_agent import MarketAgent
>>> agent = MarketAgent(company_name="어딩")
>>> result = agent.get_market_result()
>>> print(result)  # {'company_name': '어딩', 'competitor_score': 85, 'competitor_analysis_basis': '...'}

출력 스키마는 state/market_state.py 의 MarketState 형식을 따릅니다.
(필드명: company_name, competitor_score, competitor_analysis_basis)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict
from pathlib import Path

from dotenv import load_dotenv

# LangChain / LangGraph
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults

# EnsembleRetriever 경로 호환
from langchain_classic.retrievers import EnsembleRetriever

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# MarketState
from state.market_state import MarketState

# .env 로드 (프로젝트 루트)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ===== Structured Output Schema =====
class _MarketEvalSchema(BaseModel):
    startup_name: str = Field(..., description="Name of the startup under evaluation")
    market_score: int = Field(..., ge=0, le=100, description="0-100 score for market attractiveness")
    rationale: str = Field(..., description="Concise evidence-backed reasoning (Korean)")

    @field_validator("startup_name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("startup_name cannot be empty")
        return v


# ===== LangGraph State (internal) =====
class _EvalState(TypedDict):
    query: str
    startup_name: str
    retrieved_docs: List[Document]
    web_snippets: List[str]
    final_json: Optional[Dict[str, Any]]


class MarketAgent:
    def __init__(self, company_name: str, extra_hint: Optional[str] = None) -> None:
        """외부에서는 회사명만 넘기면 되도록 최소화.
        나머지 설정은 내부 기본값 또는 환경변수로 결정한다.

        환경변수(옵션):
          MARKET_COLLECTION      (기본: 'market_index')
          MARKET_CHROMA_DIR      (기본: './rag/market/chroma')
          MARKET_EMBED_MODEL     (기본: 'nomic-embed-text')
          MARKET_LLM_MODEL       (기본: 'gpt-4o-mini')
          MARKET_BM25_K          (기본: 4)
          MARKET_VS_K            (기본: 4)
          MARKET_USE_MMR         (기본: '1'이면 사용, 그 외 False)
          MARKET_ENABLE_WEB      (기본: '1'이면 활성, 그 외 False)
        """
        self.company_name = company_name or "Unknown"
        self.extra_hint = extra_hint

        # ---- 내부 기본값 + 환경변수 override ----
        self.collection_name = os.getenv("MARKET_COLLECTION", "market_index")
        self.chroma_persist_dir = os.getenv("MARKET_CHROMA_DIR", "./rag/market/chroma")
        self.embedding_model_name = os.getenv("MARKET_EMBED_MODEL", "nomic-embed-text")
        self.llm_model_name = os.getenv("MARKET_LLM_MODEL", "gpt-4o-mini")
        self.bm25_k = int(os.getenv("MARKET_BM25_K", "4"))
        self.vs_k = int(os.getenv("MARKET_VS_K", "4"))
        self.use_mmr = os.getenv("MARKET_USE_MMR", "1") == "1"
        self.tavily_enabled = os.getenv("MARKET_ENABLE_WEB", "1") == "1"

        # ---- Embeddings (Ollama) ----
        self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)

        # ---- Vector store (Chroma) ----
        self.vs = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_persist_dir,
            embedding_function=self.embeddings,
        )

        # ---- Semantic retriever ----
        if self.use_mmr:
            self.semantic_retriever = self.vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.vs_k, "fetch_k": max(20, self.vs_k * 5), "lambda_mult": 0.5},
            )
        else:
            self.semantic_retriever = self.vs.as_retriever(search_kwargs={"k": self.vs_k})

        # ---- BM25 retriever ----
        all_docs = self.vs.get(include=["metadatas", "documents"])
        bm25_docs: List[Document] = [
            Document(page_content=content or "", metadata=meta or {})
            for content, meta in zip(all_docs.get("documents", []), all_docs.get("metadatas", []))
        ]

        r_list, w_list = [self.semantic_retriever], [0.5]  # semantic weight
        if bm25_docs:
            self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            self.bm25_retriever.k = self.bm25_k
            r_list.insert(0, self.bm25_retriever)
            w_list.insert(0, 0.5)  # bm25 weight
        else:
            self.bm25_retriever = None
            print("[WARN] No docs for BM25. Using semantic retriever only.")

        self.retriever = r_list[0] if len(r_list) == 1 else EnsembleRetriever(retrievers=r_list, weights=w_list)

        # ---- LLM ----
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Set it in .env or environment.")
        self.llm = ChatOpenAI(model=self.llm_model_name, temperature=0, openai_api_key=api_key)

        # ---- Web search ----
        self.tavily = TavilySearchResults(max_results=5) if self.tavily_enabled else None

        # ---- LangGraph ----
        self.graph = self._build_graph()

    # ===== Graph =====
    def _build_graph(self):
        graph = StateGraph(_EvalState)

        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("maybe_web", self._node_maybe_web)
        graph.add_node("synthesize", self._node_synthesize)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "maybe_web")
        graph.add_edge("maybe_web", "synthesize")
        graph.add_edge("synthesize", END)

        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    # ===== Nodes =====
    def _node_retrieve(self, state: _EvalState) -> _EvalState:
        query = state["query"]
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(query)
        elif hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(query)
        else:
            docs = self.retriever._get_relevant_documents(query)  # type: ignore[attr-defined]
        state["retrieved_docs"] = docs
        return state

    def _node_maybe_web(self, state: _EvalState) -> _EvalState:
        if not self.tavily:
            state["web_snippets"] = []
            return state
        docs = state.get("retrieved_docs", [])
        total_len = sum(len(d.page_content) for d in docs)
        if len(docs) < 3 or total_len < 800:
            results = self.tavily.invoke({"query": state["query"]})
            snippets: List[str] = []
            for r in results:
                parts = [r.get("title", ""), r.get("content", "")]
                snippet = " - ".join([p for p in parts if p])
                if snippet:
                    snippets.append(snippet)
            state["web_snippets"] = snippets[:5]
        else:
            state["web_snippets"] = []
        return state

    def _node_synthesize(self, state: _EvalState) -> _EvalState:
        startup = state["startup_name"]
        docs = state.get("retrieved_docs", [])
        web_snips = state.get("web_snippets", [])

        # Context
        context_blocks: List[str] = []
        if docs:
            context_blocks.append("\n\n".join([f"[DOC {i+1}]\n" + d.page_content for i, d in enumerate(docs)]))
        if web_snips:
            context_blocks.append("\n\n".join([f"[WEB {i+1}]\n" + s for i, s in enumerate(web_snips)]))
        context_text = "\n\n".join(context_blocks) if context_blocks else "No context."

        # 출처 목록 (최대 5개)
        sources = []
        for d in docs:
            src = (d.metadata or {}).get("source")
            if src and src not in sources:
                sources.append(src)
        sources_str = ", ".join(sources[:5])

        system = (
            "You are an investment analyst specializing in AI startups. "
            "Rely on the provided context to estimate market attractiveness. "
            "Return ONLY a valid JSON object following the given schema, and write the rationale in Korean."
        )

        human_msg = (
            "Startup: {startup}\n\n"
            "Context (docs & web snippets):\n{context}\n\n"
            "Task: Assess market attractiveness and output JSON with keys:\n"
            "- startup_name (string)\n- market_score (0-100 integer)\n- rationale (<=120 words, Korean).\n\n"
            "Be concise and evidence-based."
        )
        human_msg += (
            f"\n반드시 rationale 문장 끝에 '(출처: {sources_str})'를 포함하라."
            if sources_str else "\n출처가 없으면 출처 표기는 생략하라."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", human_msg),
        ])

        chain = prompt | self.llm.with_structured_output(_MarketEvalSchema)
        out: _MarketEvalSchema = chain.invoke({"startup": startup, "context": context_text})
        state["final_json"] = out.model_dump()
        return state

    # ===== Public API =====
    def get_market_result(self) -> MarketState:
        """시장성 평가 결과를 MarketState 스키마로 반환."""
        query = self._build_query(self.company_name, hint=self.extra_hint)
        initial: _EvalState = {
            "query": query,
            "startup_name": self.company_name,
            "retrieved_docs": [],
            "web_snippets": [],
            "final_json": None,
        }
        final_state: _EvalState = self.graph.invoke(
            initial,
            config={"configurable": {"thread_id": f"market-eval-{self.company_name}"}}
        )
        payload = final_state.get("final_json") or {}

        # === 스키마 매핑 ===
        company_name = str(payload.get("startup_name") or self.company_name)
        score = int(payload.get("market_score") or 0)
        basis = str(payload.get("rationale") or "")

        result: MarketState = {
            "company_name": company_name,
            "competitor_score": score,
            "competitor_analysis_basis": basis,
        }
        return result

    # 하위 호환 (기존 이름 유지)
    def run_market_agent(self) -> MarketState:
        return self.get_market_result()

    @staticmethod
    def _build_query(startup_name: str, hint: Optional[str]) -> str:
        base = f"{startup_name} market demand, industry trend, TAM, competitors, adoption drivers"
        return base + (f"; {hint}" if hint else "")


# 하위 호환용 함수 (노드 스타일)
def analyze_market(state: MarketState) -> MarketState:
    company_name = state.get("company_name", "Unknown")
    agent = MarketAgent(company_name=company_name)
    return agent.get_market_result()


__all__ = ["MarketAgent", "analyze_market"]

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run MarketAgent (market evaluation) for a single company")
    parser.add_argument("company_name", type=str, help="회사명 (e.g., '어딩')")
    args = parser.parse_args()

    # 회사명만 전달
    agent = MarketAgent(company_name=args.company_name)
    result = agent.get_market_result()
    print(json.dumps(result, ensure_ascii=False, indent=2))
