import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)
from local_embeddings import ChromaDefaultEmbeddings


def load_documents(data_dir: Path):
    documents = []
    loaders = {
        ".pdf": lambda path: PyPDFLoader(str(path)),
        ".txt": lambda path: TextLoader(str(path), encoding="utf-8"),
    }

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in loaders:
            continue

        loaded = loaders[path.suffix.lower()](path).load()
        for doc in loaded:
            doc.metadata["source"] = path.name
            doc.metadata["source_path"] = str(path)
        documents.extend(loaded)

    return documents


def ingest(reset: bool = False) -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR.resolve()}")

    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    documents = load_documents(DATA_DIR)
    if not documents:
        raise RuntimeError(f"No PDF or TXT documents found in {DATA_DIR.resolve()}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    embeddings = ChromaDefaultEmbeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Loaded documents/pages: {len(documents)}")
    print(f"Created and stored chunks: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chroma collection: {COLLECTION_NAME}")
    print(f"Persisted at: {CHROMA_DIR.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local procurement and policy documents.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma database before ingestion.",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
