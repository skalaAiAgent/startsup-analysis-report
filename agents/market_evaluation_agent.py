"""
Market Evaluation Agent (LangGraph)

Goal:
- Given a startup name (and optional hint), assess market attractiveness by consulting a
  ChromaDB-backed RAG index (BM25 + semantic retrieval via EnsembleRetriever) and,
  if needed, augment with Tavily web search.
- Produce a structured JSON with fields: {startup_name, market_score, rationale}.

Requirements (pip):
- langchain>=0.2.0
- langgraph>=0.2.0
- langchain-openai>=0.1.0
- langchain-community>=0.2.0
- langchain-chroma>=0.1.0
- chromadb>=0.5.0

Optional:
- tavily-python (via langchain_community.tools)
- ollama running locally with an embedding model (e.g., "nomic-embed-text" or "bge-m3")

Env vars:
- OPENAI_API_KEY: for gpt-4o-mini
- TAVILY_API_KEY: for TavilySearchResults

Folder structure note:
- This file belongs to: startsup-analysis-report/agents/market_evaluation_agent.py
- It expects the market RAG index in ChromaDB (rag/market → stored as a Chroma collection).

Usage example:
>>> from agents.market_evaluation_agent import MarketEvaluator
>>> agent = MarketEvaluator(collection_name="market_index", chroma_persist_dir="./rag/market/chroma")
>>> result = agent.evaluate_startup("TripGenie AI")
>>> print(result)
{'startup_name': 'TripGenie AI', 'market_score': 78, 'rationale': '...'}

"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    # LangChain 0.2 계열에서 제공하는 shim
    from langchain_core.pydantic_v1 import BaseModel, Field, validator
except Exception:
    # 구버전 환경 호환
    from pydantic import BaseModel, Field, validator

from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults

# EnsembleRetriever import: handle version differences
try:
    from langchain.retrievers import EnsembleRetriever            # langchain 0.2.x
except Exception:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever  # 일부 배포
    except Exception:
        raise ImportError(
            "EnsembleRetriever not found. Install a compatible LangChain (e.g., langchain==0.2.16)."
        )

from langgraph.graph import StateGraph, END

from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트(startsup-analysis-report/.env) 를 찾아 로드
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# =========================
#   Output schema
# =========================
class MarketEvaluation(BaseModel):
    startup_name: str = Field(..., description="Name of the startup under evaluation")
    market_score: int = Field(..., ge=0, le=100, description="0-100 score for market attractiveness")
    rationale: str = Field(..., description="Concise evidence-backed reasoning")

    @validator("startup_name")
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("startup_name cannot be empty")
        return v


# =========================
#   LangGraph state
# =========================
class EvalState(TypedDict):
    query: str
    startup_name: str
    retrieved_docs: List[Document]
    web_snippets: List[str]
    draft_json: Optional[Dict[str, Any]]
    final_json: Optional[Dict[str, Any]]


# =========================
#   Agent class
# =========================
class MarketEvaluator:
    def __init__(
        self,
        collection_name: str,
        chroma_persist_dir: str,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "gpt-4o-mini",
        bm25_k: int = 4,
        vs_k: int = 4,
        ensemble_weights: Optional[List[float]] = None,
        tavily_enabled: bool = True,
    ) -> None:
        """Initialize the Market Evaluation Agent.

        Args:
            collection_name: Name of Chroma collection for market docs.
            chroma_persist_dir: Path to Chroma persist directory (rag/market/chroma).
            embedding_model: Ollama embedding model name.
            llm_model: OpenAI chat model name (e.g., gpt-4o-mini).
            bm25_k, vs_k: top-k for BM25 and vector search retrievers.
            ensemble_weights: weights passed to EnsembleRetriever; defaults to equal weights.
            tavily_enabled: whether to allow web search augmentation when RAG seems weak.
        """
        self.collection_name = collection_name
        self.chroma_persist_dir = chroma_persist_dir
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.bm25_k = bm25_k
        self.vs_k = vs_k
        self.ensemble_weights = ensemble_weights or [0.5, 0.5]
        self.tavily_enabled = tavily_enabled

        # --- Embeddings via Ollama ---
        self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)

        # --- Vector store (Chroma) ---
        #   Assumes documents already ingested by rag/market pipeline.
        self.vs = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_persist_dir,
            embedding_function=self.embeddings,
        )

        # --- Semantic retriever ---
        self.semantic_retriever = self.vs.as_retriever(search_kwargs={"k": self.vs_k})

        # --- BM25 retriever (in-memory) ---
        #     Build corpus dynamically from current vector store docs (metadatas & page_content)
        #     If the corpus is large, consider snapshotting BM25 index to disk in your RAG pipeline.
        all_docs = self.vs.get(include=["metadatas", "documents"])  # returns a dict
        bm25_docs: List[Document] = []
        for content, meta in zip(all_docs.get("documents", []), all_docs.get("metadatas", [])):
            bm25_docs.append(Document(page_content=content or "", metadata=meta or {}))
        self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        self.bm25_retriever.k = self.bm25_k

        # --- Ensemble retriever (existing module) ---
        self.retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.semantic_retriever],
            weights=self.ensemble_weights,
        )

        # --- LLM ---
        self.llm = ChatOpenAI(model=self.llm_model_name, temperature=0)

        # --- Web search tool (Tavily) ---
        self.tavily = TavilySearchResults(max_results=5) if self.tavily_enabled else None

        # --- Build LangGraph ---
        self.graph = self._build_graph()

    # -----------------
    # Graph definition
    # -----------------
    def _build_graph(self):
        graph = StateGraph(EvalState)

        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("maybe_web", self._node_maybe_web)
        graph.add_node("synthesize", self._node_synthesize)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "maybe_web")
        graph.add_edge("maybe_web", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    # -----------------
    # Node: retrieve
    # -----------------
    def _node_retrieve(self, state: EvalState) -> EvalState:
        query = state["query"]
        docs = self.retriever.get_relevant_documents(query)
        state["retrieved_docs"] = docs
        return state

    # -----------------
    # Node: maybe_web (heuristic)
    # -----------------
    def _node_maybe_web(self, state: EvalState) -> EvalState:
        if not self.tavily:
            state["web_snippets"] = []
            return state

        docs = state.get("retrieved_docs", [])
        # Heuristic: if fewer than 3 docs or total text < 800 chars, augment with web search
        total_len = sum(len(d.page_content) for d in docs)
        if len(docs) < 3 or total_len < 800:
            q = state["query"]
            results = self.tavily.invoke({"query": q})  # returns list of dicts
            snippets: List[str] = []
            for r in results:
                # Tavily result keys typically: url, content, title, score
                parts = [r.get("title", ""), r.get("content", "")]
                snippet = " - ".join([p for p in parts if p])
                if snippet:
                    snippets.append(snippet)
            state["web_snippets"] = snippets[:5]
        else:
            state["web_snippets"] = []
        return state

    # -----------------
    # Node: synthesize (LLM)
    # -----------------
    def _node_synthesize(self, state: EvalState) -> EvalState:
        startup = state["startup_name"]
        docs = state.get("retrieved_docs", [])
        web_snips = state.get("web_snippets", [])

        context_blocks: List[str] = []
        if docs:
            context_blocks.append("\n\n".join([f"[DOC {i+1}]\n" + d.page_content for i, d in enumerate(docs)]))
        if web_snips:
            context_blocks.append("\n\n".join([f"[WEB {i+1}]\n" + s for i, s in enumerate(web_snips)]))

        context_text = "\n\n".join(context_blocks) if context_blocks else "No context." 

        system = (
            "You are an investment analyst specializing in AI startups. "
            "Rely on the provided context to estimate market attractiveness. "
            "Return ONLY a valid JSON object following the given schema."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", (
                "Startup: {startup}\n\n"
                "Context (docs & web snippets):\n{context}\n\n"
                "Task: Assess market attractiveness and output JSON with keys: \n"
                "- startup_name (string)\n- market_score (0-100 integer)\n- rationale (<=120 words).\n\n"
                "Be concise and evidence-based."
            )),
        ])

        chain = prompt | self.llm.with_structured_output(MarketEvaluation)
        output: MarketEvaluation = chain.invoke({"startup": startup, "context": context_text})
        state["final_json"] = output.dict()
        return state

    # -----------------
    # Public API
    # -----------------
    def evaluate_startup(self, startup_name: str, extra_hint: Optional[str] = None) -> Dict[str, Any]:
        """Run the graph and return the structured evaluation.

        Args:
            startup_name: target startup (e.g., "TripGenie AI").
            extra_hint: optional hint appended to the retrieval query (e.g., domain, product).
        """
        query = self._build_query(startup_name, extra_hint)
        initial: EvalState = {
            "query": query,
            "startup_name": startup_name,
            "retrieved_docs": [],
            "web_snippets": [],
            "draft_json": None,
            "final_json": None,
        }
        final_state: EvalState = self.graph.invoke(initial)
        assert final_state.get("final_json"), "No output produced by the agent."
        return final_state["final_json"]

    @staticmethod
    def _build_query(startup_name: str, hint: Optional[str]) -> str:
        base = f"{startup_name} market demand, industry trend, TAM, competitors, adoption drivers"
        return base + (f"; {hint}" if hint else "")


# --------------
# CLI convenience
# --------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run Market Evaluation Agent")
    parser.add_argument("startup_name", type=str, help="Startup to evaluate")
    parser.add_argument("--collection", type=str, default="market_index")
    parser.add_argument("--persist_dir", type=str, default="./rag/market/chroma")
    parser.add_argument("--hint", type=str, default=None)
    parser.add_argument("--no_web", action="store_true", help="Disable Tavily web augmentation")
    args = parser.parse_args()

    agent = MarketEvaluator(
        collection_name=args.collection,
        chroma_persist_dir=args.persist_dir,
        tavily_enabled=not args.no_web,
    )
    result = agent.evaluate_startup(args.startup_name, extra_hint=args.hint)
    print(json.dumps(result, ensure_ascii=False, indent=2))
