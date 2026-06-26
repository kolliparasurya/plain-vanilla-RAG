# Plain-Vanilla RAG System

A local Retrieval-Augmented Generation system for answering questions over procurement and policy PDFs. It loads the corpus from `data/`, embeds chunks with Google `models/text-embedding-004`, stores them in a persistent Chroma database, retrieves the top 5 chunks for a question, and asks Gemini to produce a cited answer.

## How It Works

```text
data/ PDFs
  -> ingest.py
  -> RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
  -> GoogleGenerativeAIEmbeddings (models/text-embedding-004)
  -> ChromaDB persistent store in chroma_db/
  -> query.py retriever (top 5 chunks)
  -> ChatGoogleGenerativeAI (gemini-2.0-flash, temperature 0)
  -> cited answer or fixed refusal
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

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

## Key Choices

| Choice | Value | Reason |
| --- | --- | --- |
| Loader | `PyPDFLoader` | The dataset is PDF-based and page metadata is useful for traceability. |
| Chunking | `RecursiveCharacterTextSplitter`, `1000/150` | Keeps chunks large enough for policy context while preserving continuity across boundaries. |
| Embeddings | `models/text-embedding-004` | Strong Google embedding model with 768-dimensional vectors and good semantic retrieval quality. |
| Vector store | ChromaDB | Local, persistent, zero-infrastructure storage. |
| Top-k | `5` | Gives enough breadth for multi-document questions without flooding the prompt. |
| LLM | `gemini-2.0-flash` | Fast and reliable for deterministic instruction-following. |
| Temperature | `0` | Policy QA should be factual and repeatable. |

## Citations and Unanswerable Questions

Every retrieved chunk is formatted with its source filename and page number before being sent to Gemini. The system prompt requires answers to use only the provided context and to list sources under a `Sources:` heading.

If the retrieved context is insufficient, the model is instructed to output exactly:

```text
I cannot answer this based on the provided documents.
```

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
