"""
TechAgent용 PDF 인덱서
data/ 폴더의 특정 PDF 파일들을 ChromaDB에 인덱싱합니다.
"""

import os
import sys
from pathlib import Path
from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def load_pdf_documents(pdf_files: List[str]) -> List[Document]:
    """
    지정된 PDF 문서들을 로드하고 청킹
    
    Args:
        pdf_files: PDF 파일 경로 리스트
        
    Returns:
        List[Document]: 청크로 분할된 문서 리스트
    """
    print(f"\n{'='*60}")
    print(f"PDF 파일 로딩")
    print(f"{'='*60}")
    print(f"📄 대상 파일: {len(pdf_files)}개\n")
    
    all_documents = []
    loaded_files = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        pdf_path = Path(pdf_file)
        
        if not pdf_path.exists():
            print(f"  [{idx}/{len(pdf_files)}] ⚠️  파일 없음: {pdf_path.name}")
            print(f"      경로: {pdf_file}")
            continue
        
        try:
            print(f"  [{idx}/{len(pdf_files)}] 로딩 중: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # 메타데이터 추가
            for doc in documents:
                doc.metadata["source_file"] = pdf_path.name
                doc.metadata["source_type"] = "pdf"
            
            all_documents.extend(documents)
            loaded_files.append(pdf_path.name)
            print(f"      ✓ {len(documents)}페이지 로드 완료")
            
        except Exception as e:
            print(f"      ✗ PDF 로드 실패: {e}")
            continue
    
    if not all_documents:
        raise ValueError(f"로드된 PDF 파일이 없습니다. 파일 경로를 확인하세요.")
    
    print(f"\n✅ 총 {len(all_documents)}페이지 로드 완료")
    print(f"   로드된 파일: {', '.join(loaded_files)}")
    
    # 텍스트 청킹
    print(f"\n{'='*60}")
    print(f"텍스트 청킹")
    print(f"{'='*60}")
    print(f"청크 크기: 1000자")
    print(f"중복 크기: 200자\n")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    split_documents = text_splitter.split_documents(all_documents)
    print(f"✓ 총 {len(split_documents)}개의 청크 생성 완료\n")
    
    return split_documents


def build_tech_index(
    pdf_files: List[str] = None,
    chroma_persist_dir: str = None,
    collection_name: str = "startup_tech_db",
    force_rebuild: bool = False
) -> None:
    """
    지정된 PDF 파일들을 ChromaDB에 인덱싱
    
    Args:
        pdf_files: PDF 파일 경로 리스트. None이면 기본 파일들 사용
        chroma_persist_dir: ChromaDB 저장 경로. None이면 rag/tech 사용
        collection_name: ChromaDB 컬렉션 이름 (기본: startup_tech_db)
        force_rebuild: True일 경우 기존 인덱스 삭제 후 재생성
    """
    # 프로젝트 루트 경로 계산
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    data_dir = os.path.join(root, "data")
    
    # 기본 PDF 파일 목록
    if pdf_files is None:
        pdf_files = [
            os.path.join(data_dir, "기술요약_전체_기업_인터뷰.pdf"),
            os.path.join(data_dir, "시장성분석_스타트업_시장전략_및_생태계.pdf"),
            os.path.join(data_dir, "기업비교.pdf")
        ]
    
    # ChromaDB 경로 설정
    if chroma_persist_dir is None:
        chroma_persist_dir = os.path.join(root, "rag", "tech")
    
    print(f"\n{'='*60}")
    print(f"TechAgent용 PDF 인덱서")
    print(f"{'='*60}")
    print(f"프로젝트 루트: {root}")
    print(f"데이터 디렉토리: {data_dir}")
    print(f"\n대상 파일:")
    for pdf_file in pdf_files:
        exists = "✓" if os.path.exists(pdf_file) else "✗"
        print(f"  {exists} {os.path.basename(pdf_file)}")
    print(f"\nChromaDB 경로: {chroma_persist_dir}")
    print(f"컬렉션 이름: {collection_name}")
    print(f"강제 재생성: {force_rebuild}")
    print(f"{'='*60}\n")
    
    # 기존 인덱스 확인
    if os.path.exists(chroma_persist_dir) and os.path.isdir(chroma_persist_dir):
        if force_rebuild:
            print("🗑️  기존 인덱스 삭제 중...")
            import shutil
            shutil.rmtree(chroma_persist_dir)
            print("✓ 기존 인덱스 삭제 완료\n")
        else:
            print("⚠️  기존 인덱스가 존재합니다!")
            print(f"   경로: {chroma_persist_dir}")
            print(f"   기존 인덱스를 삭제하고 재생성하려면 force_rebuild=True로 설정하세요.")
            print(f"   예: build_tech_index(force_rebuild=True)\n")
            
            response = input("기존 인덱스를 삭제하고 재생성하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                print("\n❌ 인덱싱 취소됨")
                return
            
            print("\n🗑️  기존 인덱스 삭제 중...")
            import shutil
            shutil.rmtree(chroma_persist_dir)
            print("✓ 기존 인덱스 삭제 완료\n")
    
    try:
        # 1. PDF 문서 로드 및 청킹
        documents = load_pdf_documents(pdf_files)
        
        # 2. 임베딩 모델 초기화
        print(f"{'='*60}")
        print(f"임베딩 모델 초기화")
        print(f"{'='*60}")
        print(f"모델: nomic-embed-text (Ollama)\n")
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        print("✓ 임베딩 모델 초기화 완료\n")
        
        # 3. ChromaDB에 인덱싱
        print(f"{'='*60}")
        print(f"ChromaDB 인덱싱 중...")
        print(f"{'='*60}")
        print(f"⚠️  임베딩 생성 중... (수 분 소요 가능)\n")
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=chroma_persist_dir
        )
        
        print(f"\n{'='*60}")
        print(f"✅ 인덱싱 완료!")
        print(f"{'='*60}")
        print(f"저장 경로: {chroma_persist_dir}")
        print(f"컬렉션: {collection_name}")
        print(f"총 청크 수: {len(documents)}개")
        print(f"{'='*60}\n")
        
        # 4. 인덱스 테스트
        print(f"{'='*60}")
        print(f"인덱스 테스트")
        print(f"{'='*60}")
        test_query = "AI 스타트업 기술 혁신"
        print(f"테스트 쿼리: '{test_query}'\n")
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        test_docs = retriever.invoke(test_query)
        
        print(f"검색 결과: {len(test_docs)}개 문서")
        for idx, doc in enumerate(test_docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            content_preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{idx}] {source}")
            print(f"      {content_preview}...\n")
        
        print(f"{'='*60}")
        print(f"✅ 테스트 완료 - 인덱스가 정상적으로 작동합니다!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ 인덱싱 실패")
        print(f"{'='*60}")
        print(f"오류: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def verify_index(
    chroma_persist_dir: str = None,
    collection_name: str = "startup_tech_db"
) -> None:
    """
    기존 인덱스를 검증합니다.
    
    Args:
        chroma_persist_dir: ChromaDB 저장 경로. None이면 rag/tech 사용
        collection_name: ChromaDB 컬렉션 이름
    """
    # ChromaDB 경로 설정
    if chroma_persist_dir is None:
        root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        chroma_persist_dir = os.path.join(root, "rag", "tech")
    
    print(f"\n{'='*60}")
    print(f"인덱스 검증")
    print(f"{'='*60}")
    print(f"경로: {chroma_persist_dir}")
    print(f"컬렉션: {collection_name}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(chroma_persist_dir):
        print(f"❌ 인덱스가 존재하지 않습니다: {chroma_persist_dir}")
        print(f"   먼저 build_tech_index()를 실행하세요.\n")
        return
    
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_persist_dir
        )
        
        # 컬렉션 정보 확인
        collection = vectorstore._collection
        count = collection.count()
        
        print(f"✓ 인덱스 로드 성공")
        print(f"  총 청크 수: {count}개\n")
        
        # 테스트 검색
        test_queries = [
            "AI 스타트업 기술",
            "투자 평가",
            "혁신 기술"
        ]
        
        print(f"{'='*60}")
        print(f"테스트 검색")
        print(f"{'='*60}\n")
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        
        for query in test_queries:
            print(f"쿼리: '{query}'")
            docs = retriever.invoke(query)
            print(f"결과: {len(docs)}개 문서")
            
            if docs:
                doc = docs[0]
                source = doc.metadata.get("source_file", "Unknown")
                content_preview = doc.page_content[:80].replace("\n", " ")
                print(f"  상위 결과: {source}")
                print(f"  내용: {content_preview}...\n")
            else:
                print(f"  ⚠️  검색 결과 없음\n")
        
        print(f"{'='*60}")
        print(f"✅ 검증 완료 - 인덱스가 정상입니다!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ 검증 실패: {e}\n")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TechAgent용 PDF 인덱서")
    parser.add_argument(
        "--pdf-files",
        type=str,
        nargs="+",
        default=None,
        help="인덱싱할 PDF 파일 경로들 (미지정시 기본 3개 파일 사용)"
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=None,
        help="ChromaDB 저장 경로 (기본: {프로젝트}/rag/tech)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="startup_tech_db",
        help="ChromaDB 컬렉션 이름 (기본: startup_tech_db)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 인덱스 삭제 후 재생성"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="기존 인덱스 검증만 수행"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        # 검증만 수행
        verify_index(
            chroma_persist_dir=args.chroma_path,
            collection_name=args.collection
        )
    else:
        # 인덱싱 수행
        build_tech_index(
            pdf_files=args.pdf_files,
            chroma_persist_dir=args.chroma_path,
            collection_name=args.collection,
            force_rebuild=args.force
        )


if __name__ == "__main__":
    main()