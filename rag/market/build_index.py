"""
RAG Market Index Builder for ChromaDB (Ollama Embeddings)

- Loads market-analysis PDFs
- Splits into chunks
- Generates embeddings via Ollama
- Persists into a Chroma collection used by MarketEvaluationAgent

Requirements (pip):
  langchain>=0.2.0
  langchain-community>=0.2.0
  langchain-chroma>=0.1.0
  chromadb>=0.5.0

Runtime requirements:
  - Ollama running locally with an embedding model (default: "nomic-embed-text")

Example:
  python rag/market/build_index.py \
    --collection market_index \
    --persist_dir ./rag/market/chroma \
    --files "./data/시장성분석_스타트업_시장전략_및_생태계.pdf" "./data/기업비교.pdf" \
    --chunk_size 1200 --chunk_overlap 150
"""
from __future__ import annotations

import argparse
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma


def load_pdfs(paths: List[str]):
    docs = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] File not found, skipping: {p}")
            continue
        try:
            loader = PyPDFLoader(p)
            pdf_docs = loader.load()
            # annotate basic metadata
            for d in pdf_docs:
                d.metadata = d.metadata or {}
                d.metadata.setdefault("source", os.path.basename(p))
                d.metadata.setdefault("category", "market")
            docs.extend(pdf_docs)
            print(f"[INFO] Loaded {len(pdf_docs)} pages from {p}")
        except Exception as e:
            print(f"[ERROR] Failed to load {p}: {e}")
    return docs


def split_docs(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ". ", ".", " "]
    )
    splits = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(splits)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return splits


def build_chroma(collection_name: str, persist_dir: str, docs, embedding_model: str):
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = OllamaEmbeddings(model=embedding_model)
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]

    # Optional: avoid duplicate ids by letting Chroma auto-assign
    added_ids = vs.add_texts(texts=texts, metadatas=metas)
    print(f"[INFO] Added {len(added_ids)} chunks into collection '{collection_name}' @ {persist_dir}")

    # Count
    store_info = vs.get()
    print(f"[INFO] Collection now has {len(store_info.get('ids', []))} vectors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB index for market docs (Ollama embeddings)")
    parser.add_argument("--collection", default="market_index", help="Chroma collection name")
    parser.add_argument("--persist_dir", default="./rag/market/chroma", help="Chroma persist directory")
    parser.add_argument("--files", nargs="*", default=[], help="PDF files to index")
    parser.add_argument("--embedding_model", default="nomic-embed-text", help="Ollama embedding model name")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    args = parser.parse_args()

    # If no explicit files passed, fall back to commonly used project paths
    default_candidates = [
        "./data/시장성분석_스타트업_시장전략_및_생태계.pdf",
        "./data/스타트업_시장성_평가_RAG_v4.pdf",
    ]
    file_list = args.files if args.files else default_candidates

    raw_docs = load_pdfs(file_list)
    if not raw_docs:
        print("[ERROR] No documents loaded. Exiting.")
        raise SystemExit(1)

    chunks = split_docs(raw_docs, args.chunk_size, args.chunk_overlap)
    build_chroma(args.collection, args.persist_dir, chunks, args.embedding_model)

    print("[DONE] Chroma index build complete.")
