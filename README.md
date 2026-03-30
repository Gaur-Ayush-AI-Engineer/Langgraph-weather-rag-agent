# AI Chat Assistant with LangGraph & RAG

An intelligent chat assistant built with LangGraph that routes queries between real-time weather information and PDF document Q&A using Retrieval-Augmented Generation (RAG). Includes a full RAGAS evaluation pipeline with pluggable chunking strategies and query strategies.

---

## Architecture

```
User Query → Intent Classification (LangGraph) → Route Decision
                                                   ↓
                                   ┌───────────────┴────────────────┐
                                   ↓                                ↓
                            Weather Path                      Document Path
                                   ↓                                ↓
                          Extract City → Fetch API        Query Strategy (optional)
                                   ↓                                ↓
                                   │                       Vector Search (Qdrant)
                                   │                                ↓
                                   │                    Flashrank Reranker (top 3)
                                   │                                ↓
                                   └────────→ LLM Response ←────────┘
```

---

## Project Structure

```
project/
├── agent.py                  # LangGraph agent pipeline
├── weather.py                # OpenWeatherMap API integration
├── rag.py                    # RAG tool — Qdrant, chunking, reranking, query strategies
├── chunkers.py               # StructureAwareChunker (section-based PDF splitting)
├── query_strategies.py       # QueryRewriter and MultiQueryRetriever
├── database.py               # SQLite chat history
├── app.py                    # Streamlit UI
├── evaluate_rag.py           # RAGAS evaluation script — sweeps query strategies
├── eval_data/
│   ├── attention_is_all_you_need.pdf   # PDF used for evaluation
│   └── golden_dataset.json             # 10 Q&A pairs for RAGAS evaluation
├── tests/
│   ├── test_weather.py
│   ├── test_rag.py
│   ├── test_agent.py
│   └── test_database.py
├── requirements.txt
├── .env
└── README.md
```

---

## Key Components

### `rag.py` — RAGTool
Central class. Constructor signature:
```python
RAGTool(
    chunk_size=1000,
    chunk_overlap=200,
    chunker=None,              # pass StructureAwareChunker to override default splitting
    rerank_threshold=0.0,      # drop reranked chunks below this score (0.0 = keep all top-3)
    query_strategy=None        # pass QueryRewriter or MultiQueryRetriever
)
```

**RAG flow:**
1. `load_pdf(path)` — ingests PDF into a Qdrant collection named `pdf_<name>_<chunker_tag>`. Skips ingestion if collection already exists.
2. `query(question)` — applies query strategy → retrieves top 5 chunks from Qdrant → Flashrank reranker keeps top 3 (filtered by threshold) → GPT-4o-mini generates answer.
3. Returns `{"answer": ..., "sources": [...], "contexts": [...]}`. `contexts` contains full chunk text (required by RAGAS).

**Collection naming:** each unique combination of PDF + chunker type gets its own Qdrant collection, so re-ingestion only happens once per config.

---

### `chunkers.py` — StructureAwareChunker
Splits PDFs by document structure instead of fixed character counts.

**How it works:**
1. Detects numbered section headers via regex (e.g. `3.1 Encoder and Decoder Stacks`)
2. Splits the full document into per-section blocks
3. Sub-splits large sections at paragraph boundaries (`\n\n`), never mid-sentence
4. Prefixes every chunk with `[Section X.Y: Title]` so the LLM always has section context

```python
from chunkers import StructureAwareChunker
chunker = StructureAwareChunker(max_chunk_size=1000, chunk_overlap=100)
rag = RAGTool(chunker=chunker)
```

If no chunker is passed, `RAGTool` falls back to LangChain's `RecursiveCharacterTextSplitter`.

---

### `query_strategies.py` — QueryRewriter & MultiQueryRetriever

**QueryRewriter** — rewrites the user's question before retrieval:
- Calls GPT-4o-mini to produce a more document-friendly version of the query
- Uses 1 rewritten query → Qdrant returns top 5 → reranker keeps top 3
- Logs original and rewritten query to console

**MultiQueryRetriever** — expands the query into multiple variants:
- Calls GPT-4o-mini to generate 3 alternative phrasings
- Runs Qdrant retrieval for each variant (original + 3 = 4 queries)
- Deduplicates results by content fingerprint
- Passes the larger unique pool to the reranker, which still keeps top 3
- Gives the reranker more diverse candidates → better context precision

```python
from query_strategies import QueryRewriter, MultiQueryRetriever

rag = RAGTool(query_strategy=QueryRewriter())
rag = RAGTool(query_strategy=MultiQueryRetriever(n_variants=3))
```

---

### `evaluate_rag.py` — RAGAS Evaluation
Standalone script that evaluates the RAG pipeline using [RAGAS](https://docs.ragas.io).

**Golden dataset:** 10 Q&A pairs in `eval_data/golden_dataset.json` covering:
- Architecture (encoder/decoder layers, attention heads, FFN dimensions)
- Key concepts (scaled dot-product attention, positional encoding)
- Training details (Adam optimizer, learning rate schedule)
- Results (BLEU scores, WMT 2014 datasets)

**What it does:**
1. Loads the PDF (ingests into Qdrant if collection doesn't exist)
2. Runs all 10 questions through the RAG pipeline for each strategy
3. Evaluates with 4 RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
4. Saves per-strategy JSON results and prints a comparison table

**Config block at the top of `evaluate_rag.py`:**
```python
USE_STRUCTURE_AWARE_CHUNKING = True   # use StructureAwareChunker
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
RERANK_THRESHOLD = 0.0                # best value from threshold sweep
STRATEGIES       = ["none", "rewrite", "multi"]  # strategies to compare
```

**Output files** (one per strategy):
```
eval_results_struct_none.json
eval_results_struct_rewrite.json
eval_results_struct_multi.json
```

**Run:**
```bash
python evaluate_rag.py
```

**Actual results (structure-aware chunking, chunk_size=1000, rerank_threshold=0.0):**

| Metric | none | rewrite | multi |
|---|---|---|---|
| faithfulness | **0.9104** | NaN* | 0.8917 |
| answer_relevancy | 0.8628 | 0.8726 | **0.8740** |
| context_precision | 0.6667 | 0.6667 | **0.6750** |
| context_recall | **0.8000** | 0.7500 | 0.7500 |

**Overall winner: `none`** — best faithfulness and context recall, no metric failures.
**`multi` wins narrowly on context_precision** (0.675 vs 0.667).

*`rewrite` strategy returned NaN for faithfulness — RAGAS metric computation failed for that strategy, likely a context formatting issue. Treat rewrite faithfulness scores as invalid.

---

## RAG Tuning Findings

| What was tested | Result |
|---|---|
| Chunk size 500 vs 1000 | 500 hurt context recall — concepts split across boundaries. **Kept 1000.** |
| Fixed-size vs structure-aware chunking | Structure-aware keeps section content together. **Better overall.** |
| Rerank threshold (0.0 / 0.1 / 0.2 / 0.3 / 0.5) | Any filtering dropped useful chunks. **Best: 0.0 (keep all top-3).** |
| Query strategies (none / rewrite / multi) | Minimal improvement on context_precision (+0.008 max). **Not worth LLM cost at scale.** |

**Context precision is the ceiling.** All strategies plateaued at 0.667–0.675. This is a retrieval-side issue — query strategies alone can't fix it. Improving embeddings or the Qdrant index configuration would be the next lever.

---

## RAGAS Metrics Explained

| Metric | What it measures | Low score means |
|---|---|---|
| `faithfulness` | Are answers grounded in retrieved context? | LLM is hallucinating |
| `answer_relevancy` | Does the answer address the question? | Answer is off-topic |
| `context_precision` | Are retrieved chunks relevant? | Too much noise in retrieval |
| `context_recall` | Do retrieved chunks contain the needed info? | Missing relevant chunks |

---

## Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- OpenAI API key
- OpenWeatherMap API key
- LangSmith API key (optional)

---

## Installation

```bash
# 1. Clone
git clone <your-repo-url>
cd <project-directory>

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 5. Configure environment
cp .env.example .env
# Edit .env:
# OPENAI_API_KEY=sk-...
# OPENWEATHERMAP_API_KEY=...
# LANGCHAIN_TRACING_V2=true       (optional)
# LANGCHAIN_API_KEY=lsv2_pt_...   (optional)
# LANGCHAIN_PROJECT=...           (optional)
```

---

## Running the App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

1. Upload a PDF via the sidebar
2. Ask weather questions: `"What's the weather in Tokyo?"`
3. Ask document questions: `"What are the main findings?"`

---

## Running Evaluation

```bash
# Make sure Qdrant is running and attention_is_all_you_need.pdf is in eval_data/
python evaluate_rag.py
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

---

## RAG Flow Summary

```
question
   ↓
[Query Strategy]
   ├── none    → use question as-is
   ├── rewrite → LLM rewrites question → 1 improved query
   └── multi   → LLM generates 3 variants → 4 queries → deduplicated pool
   ↓
Qdrant vector search (top 5 per query)
   ↓
Flashrank reranker (ms-marco-MiniLM-L-12-v2) → keeps top 3 above threshold
   ↓
GPT-4o-mini answer generation with top 3 chunks as context
   ↓
{"answer": ..., "sources": [...], "contexts": [...]}
```

---

## Database Schema

```sql
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_query TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    intent TEXT,
    pdf_name TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now'))
)
```

---

## Known Limitations

- PDFs must be re-uploaded if Qdrant data is not persisted (use `-v` mount flag with Docker)
- Single-user design (no authentication)
- Structure-aware chunker regex is tuned for numbered academic paper sections

---

## Author

Ayush Gaur — ayushgaur228@gmail.com
