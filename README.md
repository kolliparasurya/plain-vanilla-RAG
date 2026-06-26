# Plain-Vanilla RAG System

A local Retrieval-Augmented Generation system for answering questions over procurement and policy PDFs. It loads the corpus from `data/`, embeds chunks with ChromaDB's built-in local default embedding function, stores them in a persistent Chroma database, retrieves the top 5 chunks for a question, and can ask Gemini to produce a cited answer when `GOOGLE_API_KEY` is available.

## How the Pipeline Works

The system has three main stages: ingestion, retrieval/generation, and evaluation.

1. `ingest.py` loads every PDF from `data/` using `PyPDFLoader`. Each page keeps metadata such as the source filename and page number, which later becomes citation information.
2. The loaded documents are split with `RecursiveCharacterTextSplitter` into 1000-character chunks with 150 characters of overlap. `add_start_index=True` is enabled so each chunk remains traceable to its original position.
3. Each chunk is embedded locally with ChromaDB's built-in default embedding function.
4. The chunks and embeddings are stored in a local persistent ChromaDB collection at `chroma_db/`. Ingestion adds chunks in batches and pauses between batches to stay within free-tier embedding rate limits.
5. `query.py` loads the same ChromaDB collection, embeds the user question with the same embedding model, and retrieves the top 5 most relevant chunks.
6. Retrieved chunks are formatted as context blocks with source filenames and page numbers.
7. Gemini `gemini-2.0-flash` receives the system prompt, retrieved context, and user question. It must answer only from the provided context, cite sources, or return the fixed refusal message.
8. `evaluate.py` can run sample questions and use Gemini as a judge to check whether the generated answer is faithful to the retrieved context.

```text
data/ PDFs
  -> ingest.py
  -> PDF pages with source metadata
  -> RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=150)
  -> ChromaDB DefaultEmbeddingFunction (local embeddings)
  -> ChromaDB persistent vector store (chroma_db/)
  -> query.py retrieves top-k chunks (k=5)
  -> prompt = system rules + retrieved context + user question
  -> ChatGoogleGenerativeAI (gemini-2.0-flash, temperature=0)
  -> cited answer or fixed refusal
```

## Setup

```bash
pip install -r requirements.txt
```

For Gemini answer generation, create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

Ingestion and retrieval do not need this key because embeddings are local.

The repository expects PDFs to be present in `data/`. The current corpus includes DPM 2025 volumes, Delegation of Financial Powers Rules 2024, and Navy regulations PDFs.

## Run

Ingest the corpus once:

```bash
python ingest.py --reset
```

Ask a single question:

```bash
python query.py --query "What is the policy on procurement thresholds?"
```

Retrieve chunks without calling Gemini:

```bash
python query.py --query "What is the policy on procurement thresholds?" --retrieve-only
```

Start an interactive session:

```bash
python query.py --interactive
```

Run faithfulness evaluation:

```bash
python evaluate.py
```

You can also evaluate custom questions with one question per line:

```bash
python evaluate.py --questions-file questions.txt
```

## Key Choices and Why

| Choice | Value | Reason |
| --- | --- | --- |
| Loader | `PyPDFLoader` | The dataset is PDF-based and page metadata is useful for traceability. |
| Chunking | `RecursiveCharacterTextSplitter`, `chunk_size=1000`, `chunk_overlap=150` | Recursive splitting respects natural text boundaries better than a blind fixed-width split. A 1000-character chunk is large enough to hold a policy clause or short section, while 150 characters of overlap reduces the chance that an answer is split across two chunks. |
| Chunk metadata | `source`, `source_path`, `page`, `start_index` | Source and page metadata make citations possible. Start indexes make chunks easier to inspect during debugging. |
| Embedding model | ChromaDB `DefaultEmbeddingFunction` | Embeddings run locally through ChromaDB, so ingestion and retrieval do not require a Gemini API key or external embedding quota. |
| Vector store | ChromaDB | Local, persistent, zero-infrastructure storage. |
| Vector store path | `chroma_db/` | The database is written to disk once during ingestion and reused at query time, so documents do not need to be re-embedded for every question. |
| Retrieval `k` | `5` | Five chunks gives the LLM enough evidence for multi-document or cross-section questions while keeping the prompt focused. A much larger `k` can dilute the context with weak matches. |
| LLM | `gemini-2.0-flash` | Fast and reliable for deterministic instruction-following. |
| Temperature | `0` | Policy QA should be factual and repeatable. |
| Prompt strategy | Strict context-only system prompt | The prompt tells the model to use only retrieved context, avoid outside knowledge, cite source documents, and refuse when the answer is unsupported. This is the main guardrail against hallucinated policy answers. |

## Citations and Unanswerable Questions

Every retrieved chunk is formatted with its source filename and page number before being sent to Gemini:

```text
[Source: DPM-2025-VOLUME-I.pdf, page 12]
<chunk text>
```

The system prompt enforces three rules:

1. Answer only using the provided retrieved context.
2. Do not use outside knowledge or training data.
3. List the source document names under a `Sources:` heading when an answer is supported.

If the retrieved context is insufficient, the model is instructed to output exactly:

```text
I cannot answer this based on the provided documents.
```

This fixed refusal string makes unanswerable handling easy to test automatically. `temperature=0` is used to make the refusal behavior more consistent, and `evaluate.py` includes an out-of-corpus sample question to check whether the system refuses instead of guessing.

## Evaluation

`evaluate.py` implements faithfulness judging with Gemini as an evaluator. For each sample question, the pipeline generates an answer, passes the retrieved context and answer to a judge prompt, and records `YES` only when every claim is explicitly supported by the retrieved context.

Metrics to track:

| Metric | What it measures | How |
| --- | --- | --- |
| Context hit rate | Whether the answer is present in top-k chunks | Inspect retrieved chunks for expected terms or answers. |
| Faithfulness | Whether the answer stays within the retrieved context | Gemini-as-judge in `evaluate.py`. |
| Unanswerable accuracy | Whether the system refuses unsupported questions | Include out-of-corpus questions in evaluation. |

Paste evaluation results here after running `python evaluate.py`:

```text
YES: pending
NO: pending
Total: pending
```

## Improvements With More Time

- Semantic chunking instead of character-count splitting.
- Cross-encoder re-ranking of the retrieved chunks.
- Hybrid retrieval combining BM25 keyword search with vector similarity.
- Stronger citation validation by mapping each sentence to source chunks.
- RAGAS or another automated evaluation framework for richer measurement.
