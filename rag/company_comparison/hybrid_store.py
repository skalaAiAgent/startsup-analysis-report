from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from chromadb import PersistentClient
except Exception:
    PersistentClient = None


_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase alphanumerics and dashes only."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


class HybridStore:
    """BM25 + Chroma hybrid retriever with HuggingFace embeddings.

    - Stores evidence chunks in both BM25 and Chroma with shared ids/metadata.
    - Embeddings: jinaai/jina-embeddings-v2-base-ko (normalized).
    - Search: Reciprocal Rank Fusion (RRF) + optional lambda-weighted sum of normalized scores.
    - No SQLite; all evidence is from PDF text chunks.
    """

    def __init__(self, chroma_path: str, collection: str = "company_comparison") -> None:
        self.chroma_path = os.path.normpath(chroma_path)
        self.collection_name = collection

        # In-memory BM25 index
        self._bm25_tokens: List[List[str]] = []
        self._bm25_ids: List[str] = []
        self._bm25_texts: List[str] = []
        self._bm25_metas: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None

        # Chroma persistent collection
        if PersistentClient is None:
            raise RuntimeError("chromadb is required but not installed.")
        os.makedirs(self.chroma_path, exist_ok=True)
        self._client = PersistentClient(path=self.chroma_path)
        
        # 컬렉션 가져오기 또는 생성
        try:
            self._coll = self._client.get_or_create_collection(name=self.collection_name)
        except Exception:
            self._coll = self._client.create_collection(name=self.collection_name)

        # Embedding model (lazy)
        self._model: Optional[SentenceTransformer] = None
        
        # 기존 데이터 로드 (BM25용)
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """기존 Chroma DB의 데이터를 BM25 인덱스에 로드합니다."""
        try:
            # Chroma에서 모든 데이터 가져오기
            results = self._coll.get(include=["documents", "metadatas"])
            
            if results and results.get("documents"):
                docs = results["documents"]
                ids = results["ids"]
                metas = results.get("metadatas", [{}] * len(docs))
                
                # BM25 인덱스 구축
                self._bm25_texts = docs
                self._bm25_ids = ids
                self._bm25_metas = metas
                self._bm25_tokens = [_tokenize(text) for text in docs]
                
                if self._bm25_tokens:
                    self._bm25 = BM25Okapi(self._bm25_tokens)
                
                print(f"✅ 기존 인덱스 로드 완료: {len(docs)}개 문서")
        except Exception as e:
            print(f"⚠️ 기존 데이터 로드 실패 (신규 인덱스 생성): {e}")

    # -------------------------------
    # Embedding
    # -------------------------------
    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            print("🔄 임베딩 모델 로딩 중: jhgan/ko-sroberta-multitask")
            self._model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            print("✅ 임베딩 모델 로드 완료")
        return self._model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Encode texts with SentenceTransformers, normalized embeddings."""
        if not texts:
            return []
        model = self._ensure_model()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=True,
        )
        return vecs.tolist()

    # -------------------------------
    # Indexing
    # -------------------------------
    def add_evidence_chunks(
        self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]
    ) -> None:
        """Add evidence to BM25 and Chroma using shared ids/metadata.

        Args:
            texts: list of text chunks
            metadatas: list of metadata dicts (JSON-serializable)
            ids: stable ids for each chunk
        """
        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError("texts, metadatas, ids must have same length")

        # Update BM25
        toks = [_tokenize(t) for t in texts]
        self._bm25_tokens.extend(toks)
        self._bm25_ids.extend(ids)
        self._bm25_texts.extend(texts)
        self._bm25_metas.extend(metadatas)
        self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None

        # Add to Chroma
        print("🔄 임베딩 생성 중...")
        embeddings = self._embed(texts)
        
        # Ensure metadata is serializable
        safe_metas: List[Dict[str, Any]] = []
        for m in metadatas:
            safe: Dict[str, Any] = {}
            for k, v in (m or {}).items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    safe[k] = v
                else:
                    safe[k] = str(v)
            safe_metas.append(safe)
        
        print("🔄 Chroma DB에 저장 중...")
        self._coll.add(ids=ids, documents=texts, metadatas=safe_metas, embeddings=embeddings)
        print("✅ 저장 완료")

    # -------------------------------
    # Search
    # -------------------------------
    def search(
        self,
        query: str,
        k_bm25: int = 8,
        k_vec: int = 8,
        rrf_k: int = 60,
        lambda_vec: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining BM25 and vector results.

        Returns list of dicts: {doc_id, text, metadata, score}
        """
        bm25_pairs: List[Tuple[str, float]] = []
        vec_pairs: List[Tuple[str, float]] = []

        # BM25
        if self._bm25 is not None and self._bm25_tokens:
            q_tokens = _tokenize(query)
            scores = self._bm25.get_scores(q_tokens)
            idx = np.argsort(scores)[::-1][: max(1, k_bm25)]
            bm25_pairs = [(self._bm25_ids[i], float(scores[i])) for i in idx]

        # Vector via Chroma
        q_vec = self._embed([query])[0]
        try:
            out = self._coll.query(
                query_embeddings=[q_vec],
                n_results=max(1, k_vec),
                include=["metadatas", "documents", "distances"],
            )
            if out and out.get("ids"):
                ids = out["ids"][0]
                dists = out.get("distances", [[]])[0]
                # Convert distance to similarity (negative distance)
                vec_pairs = [(str(i), -float(d)) for i, d in zip(ids, dists)]
        except Exception as e:
            print(f"⚠️ Vector search error: {e}")
            vec_pairs = []

        # Normalize for weighted sum
        def _normalize(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
            if not pairs:
                return {}
            vals = np.array([s for _, s in pairs], dtype=float)
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax - vmin < 1e-9:
                return {i: 0.5 for i, _ in pairs}
            return {i: (s - vmin) / (vmax - vmin) for i, s in pairs}

        bm25_norm = _normalize(bm25_pairs)
        vec_norm = _normalize(vec_pairs)

        # RRF scores
        def _rrf(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
            ranks: Dict[str, int] = {}
            for r, (i, _) in enumerate(
                sorted(pairs, key=lambda x: x[1], reverse=True), start=1
            ):
                ranks[i] = r
            return {i: 1.0 / (rrf_k + r) for i, r in ranks.items()}

        rrf_bm25 = _rrf(bm25_pairs)
        rrf_vec = _rrf(vec_pairs)

        # Fuse scores
        fused: Dict[str, float] = {}
        all_ids = set([i for i, _ in bm25_pairs] + [i for i, _ in vec_pairs])
        for i in all_ids:
            score = rrf_bm25.get(i, 0.0) + rrf_vec.get(i, 0.0)
            if lambda_vec != 0.0:
                score += lambda_vec * (bm25_norm.get(i, 0.0) + vec_norm.get(i, 0.0))
            fused[i] = score

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Build results
        id2text = {i: t for i, t in zip(self._bm25_ids, self._bm25_texts)}
        id2meta = {i: m for i, m in zip(self._bm25_ids, self._bm25_metas)}

        results: List[Dict[str, Any]] = []
        seen: set = set()
        
        for i, s in ranked:
            if i in seen:
                continue
            seen.add(i)
            
            text = id2text.get(i)
            meta = id2meta.get(i, {})
            
            # Fallback to Chroma if not in BM25 cache
            if text is None:
                try:
                    out = self._coll.get(ids=[i], include=["documents", "metadatas"])
                    if out and out.get("documents"):
                        text = out["documents"][0]
                        meta = out.get("metadatas", [{}])[0]
                except Exception:
                    text = ""
                    meta = {}
            
            if text is None:
                text = ""
            
            results.append({
                "doc_id": i,
                "text": text,
                "metadata": meta,
                "score": float(s)
            })
        
        return results


__all__ = ["HybridStore"]