# CLAUDE.md — Instructions for Claude Code

This file tells Claude how to work in this repo without re-discovering things from scratch.

---

## What this repo is

A LangGraph-based chat assistant that routes queries to either a weather API or a RAG pipeline. It also has a full RAGAS evaluation pipeline for the RAG component, built around the "Attention Is All You Need" paper.

---

## Critical rules

- **Never rewrite RAG logic inline.** All RAG logic lives in `rag.py`. Import `RAGTool` from there.
- **Never add new chunkers inline.** All chunkers go in `chunkers.py` and must implement a `split_documents(documents) -> List[Document]` method.
- **Never add new query strategies inline.** All query strategies go in `query_strategies.py`. `QueryRewriter` has a `rewrite(question) -> str` method. `MultiQueryRetriever` has `expand(question) -> List[str]` and `retrieve_and_deduplicate(question, retriever, k)`.
- **Never manually create Qdrant collections.** `RAGTool.load_pdf()` handles this automatically. Collection names encode the PDF name + chunker type so each config gets its own collection.
- **Qdrant must be running on port 6333** before any RAG or evaluation code can run. Start it with `docker run -p 6333:6333 qdrant/qdrant`.

---

## Key files and their roles

| File | Role |
|---|---|
| `rag.py` | Central RAG class. Handles ingestion, retrieval, reranking, answer generation |
| `chunkers.py` | `StructureAwareChunker` — splits PDFs by section headers + paragraphs |
| `query_strategies.py` | `QueryRewriter`, `MultiQueryRetriever` — applied before Qdrant search |
| `evaluate_rag.py` | RAGAS evaluation script — sweeps query strategies, prints comparison table |
| `eval_data/golden_dataset.json` | 10 Q&A pairs about "Attention Is All You Need" — the evaluation ground truth |
| `agent.py` | LangGraph pipeline — intent classification + routing |
| `app.py` | Streamlit UI |
| `database.py` | SQLite chat history |
| `weather.py` | OpenWeatherMap API integration |

---

## RAGTool constructor

```python
RAGTool(
    chunk_size=1000,           # used only when chunker=None
    chunk_overlap=200,         # used only when chunker=None
    chunker=None,              # pass StructureAwareChunker() to override splitting
    rerank_threshold=0.0,      # reranker score cutoff; 0.0 = keep all top-3
    query_strategy=None        # pass QueryRewriter() or MultiQueryRetriever()
)
```

`rag.query(question)` returns:
```python
{
    "answer": str,
    "sources": [...],          # truncated (200 chars) — for display only
    "contexts": [str, ...]     # full chunk text — this is what RAGAS needs
}
```

Always use `result["contexts"]` for RAGAS, not `result["sources"]`.

---

## Evaluation pipeline

**Config lives at the top of `evaluate_rag.py`:**
```python
USE_STRUCTURE_AWARE_CHUNKING = True
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
RERANK_THRESHOLD = 0.0          # best value from prior threshold sweep
STRATEGIES       = ["none", "rewrite", "multi"]
```

**To run:**
```bash
python evaluate_rag.py
```

**Output files** (one per strategy, auto-named):
```
eval_results_struct_none.json
eval_results_struct_rewrite.json
eval_results_struct_multi.json
```

Each file contains `aggregate_scores` and `per_question` breakdown.

**RAGAS metrics:**
- `faithfulness` — answers grounded in context? Low = hallucination
- `answer_relevancy` — answer addresses the question? Low = off-topic
- `context_precision` — retrieved chunks relevant? Low = noisy retrieval
- `context_recall` — retrieved chunks contain needed info? Low = missing chunks

---

## RAG flow (with all options active)

```
question
   ↓
[Query Strategy]
   ├── none    → question used as-is
   ├── rewrite → GPT-4o-mini rewrites → 1 improved query
   └── multi   → GPT-4o-mini generates 3 variants → 4 queries total
                 → retrieve top 5 per query → deduplicate by content fingerprint
   ↓
Qdrant cosine search (top 5)
   ↓
Flashrank reranker (ms-marco-MiniLM-L-12-v2)
   → drops chunks below rerank_threshold
   → keeps top 3 (fallback to top-1 if all below threshold)
   ↓
GPT-4o-mini — answer generation using top 3 chunks as context
```

---

## Qdrant collection naming

Collection name = `pdf_{sanitized_pdf_name}_{chunker_tag}`

- `chunker_tag` = `struct` when `StructureAwareChunker` is used
- `chunker_tag` = `c{chunk_size}` when default splitter is used (e.g. `c1000`)

This means ingestion only happens once per unique PDF + chunker combo. Changing only `rerank_threshold` or `query_strategy` reuses the existing collection.

---

## What was tuned and why

- **Chunk size 1000 → 500**: hurt context recall (chunks too small, concepts split across chunks)
- **Structure-aware chunking**: introduced to keep section content together; better than fixed-size
- **Rerank threshold sweep** (0.0, 0.1, 0.2, 0.3, 0.5): best was `0.0` — filtering dropped useful chunks
- **Query strategies**: added to improve `context_precision` (was ~0.67), which was the weakest metric

---

## Environment variables (.env)

```
OPENAI_API_KEY=sk-...
OPENWEATHERMAP_API_KEY=...
LANGCHAIN_TRACING_V2=true        # optional
LANGCHAIN_API_KEY=lsv2_pt_...   # optional
LANGCHAIN_PROJECT=...            # optional
```

---

## Running tests

```bash
pytest tests/ -v
```

All tests mock external services (Qdrant, OpenAI, OpenWeatherMap) — no live API calls needed.
