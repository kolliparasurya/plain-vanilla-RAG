import argparse
from typing import Iterable, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import (
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    REFUSAL_MESSAGE,
    RETRIEVAL_K,
)


SYSTEM_PROMPT = f"""You are a precise assistant for procurement and policy documents.
Answer the user's question using ONLY the context provided below.
Do not use any outside knowledge or training data.
After your answer, list the source document(s) you used under a "Sources:" heading.
If the context does not contain sufficient information to answer the question, respond exactly with: "{REFUSAL_MESSAGE}"
Do not guess or infer beyond what is written."""


def load_vector_store() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(question: str, k: int = RETRIEVAL_K) -> List[Document]:
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def format_context(chunks: Iterable[Document]) -> str:
    formatted_chunks = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        formatted_chunks.append(f"[Source: {source}{page_text}]\n{chunk.page_content}")
    return "\n\n".join(formatted_chunks)


def answer_question(question: str):
    chunks = retrieve(question)
    context = format_context(chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, chunks, context


def print_result(question: str) -> None:
    answer, chunks, _ = answer_question(question)
    print(answer)
    print("\nRetrieved chunks:")
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        start = chunk.metadata.get("start_index")
        start_text = f", start {start}" if start is not None else ""
        print(f"{index}. {source}{page_text}{start_text}")


def interactive_loop() -> None:
    print("Ask a question, or type 'exit' to quit.")
    while True:
        question = input("\nQuery> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print_result(question)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask questions over the local RAG corpus.")
    parser.add_argument("--query", help="Question to answer.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive query loop.",
    )
    args = parser.parse_args()

    if args.query:
        print_result(args.query)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
