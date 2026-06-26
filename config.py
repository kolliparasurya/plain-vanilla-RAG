from pathlib import Path

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "procurement_policy_docs"

EMBEDDING_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.0-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5

REFUSAL_MESSAGE = "I cannot answer this based on the provided documents."
