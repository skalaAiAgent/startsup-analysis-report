from __future__ import annotations

import os
import sys

# 프로젝트 루트를 Python 경로에 추가 (import 전에 실행)
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import uuid
from typing import Dict, List

from pypdf import PdfReader
from rag.company_comparison.hybrid_store import HybridStore


def build_company_comparison_index() -> None:
    """
    data/기업비교.pdf 파일을 읽어서 HybridStore에 인덱싱합니다.
    .chroma/ 폴더에 인덱스가 생성됩니다.
    """
    # 경로 설정
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    data_dir = os.path.join(root, "data")
    # 현재 Agent 폴더 안에 .chroma 생성
    chroma_dir = os.path.join(os.path.dirname(__file__), ".chroma")
    
    # PDF 파일 경로
    pdf_path = os.path.join(data_dir, "기업비교.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    print(f"📄 PDF 로딩 중: {pdf_path}")
    
    # HybridStore 초기화
    store = HybridStore(chroma_path=chroma_dir, collection="company_comparison")
    
    # PDF 읽기
    texts: List[str] = []
    metas: List[Dict[str, str]] = []
    ids: List[str] = []
    
    try:
        reader = PdfReader(pdf_path)
        print(f"📖 총 {len(reader.pages)}페이지 처리 중...")
        
        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            
            # 페이지별 텍스트를 줄 단위로 분할
            lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
            
            # 50~150 단어 단위로 더 세밀하게 청크 분할 (표 데이터 보존)
            current_chunk = []
            current_word_count = 0
            
            for line in lines:
                words = line.split()
                line_word_count = len(words)
                
                # 현재 청크에 추가
                current_chunk.append(line)
                current_word_count += line_word_count
                
                # 50 단어 이상이면 청크 저장 (더 세밀하게)
                if current_word_count >= 50:
                    chunk_text = "\n".join(current_chunk)
                    texts.append(chunk_text)
                    metas.append({
                        "source_pdf": "기업비교.pdf",
                        "page": str(page_idx + 1)
                    })
                    ids.append(str(uuid.uuid4()))
                    
                    # 청크 초기화
                    current_chunk = []
                    current_word_count = 0
                
                # 150 단어 초과 시 강제로 청크 분할
                if current_word_count >= 150:
                    chunk_text = "\n".join(current_chunk)
                    texts.append(chunk_text)
                    metas.append({
                        "source_pdf": "기업비교.pdf",
                        "page": str(page_idx + 1)
                    })
                    ids.append(str(uuid.uuid4()))
                    
                    current_chunk = []
                    current_word_count = 0
            
            # 페이지 마지막 남은 청크 저장
            if current_chunk:
                chunk_text = "\n".join(current_chunk)
                texts.append(chunk_text)
                metas.append({
                    "source_pdf": "기업비교.pdf",
                    "page": str(page_idx + 1)
                })
                ids.append(str(uuid.uuid4()))
        
        print(f"✅ 총 {len(texts)}개 청크 생성 완료")
        
        # HybridStore에 인덱싱
        print("🔄 인덱싱 중...")
        store.add_evidence_chunks(texts=texts, metadatas=metas, ids=ids)
        print(f"✅ 인덱싱 완료! ({chroma_dir})")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    build_company_comparison_index()