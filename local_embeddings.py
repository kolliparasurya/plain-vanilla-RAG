from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    """LangChain adapter for ChromaDB's built-in local embedding function."""

    def __init__(self) -> None:
        self._embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(value) for value in embedding]
            for embedding in self._embedding_function(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
