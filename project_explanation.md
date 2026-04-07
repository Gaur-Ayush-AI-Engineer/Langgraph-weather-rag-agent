# Project Explanation — Interview Guide

---

## Elevator Pitch (30 seconds)

> "I built a LangGraph-based conversational agent that intelligently routes user queries — weather questions go to a live weather API, and knowledge-based questions go through a full RAG pipeline. I also built an evaluation framework using RAGAS to benchmark different retrieval strategies."

---

## Architecture Overview

- **LangGraph pipeline** (`agent.py`) — classifies intent and routes to either a weather tool or a RAG tool
- **RAG pipeline** (`rag.py`) — ingests PDFs into Qdrant, retrieves chunks, reranks with Flashrank, generates answers using GPT-4o-mini
- **Streamlit UI** (`app.py`) with SQLite chat history (`database.py`)
- **RAGAS evaluation** (`evaluate_rag.py`) — benchmarks retrieval strategies on 10 ground-truth Q&A pairs

---

## Structure-Aware Chunking

### The problem with normal chunking
Regular chunkers cut text every N characters blindly. A sentence explaining "Multi-Head Attention" might get split in half — half in chunk 3, half in chunk 4. Retrieval then finds neither chunk useful.

### What our chunker does (3 steps)

1. **Find section headers** — Uses a regex to detect numbered headings like `"3.1 Encoder and Decoder Stacks"` — natural boundaries in academic papers
2. **Split by section first** — "Introduction" is one block, "Model Architecture" is another. No arbitrary cuts
3. **Sub-split by paragraphs only if needed** — If a section exceeds 1000 chars, it splits at blank lines (paragraph breaks), never mid-sentence. Every sub-chunk is prefixed with `[Section 3.1: Encoder and Decoder Stacks]` so the LLM always knows which section a passage came from

### One-liner
> "Instead of cutting every N characters blindly, we detect section headers with a regex, split there, and only further split at paragraph boundaries if a section is too large. Each chunk keeps its section label."

---

## Query Strategies

### The problem
A user asks *"how does the model pay attention to different words?"* — but the paper uses terms like *"scaled dot-product attention"* and *"query-key-value mechanism"*. The embedding search misses because vocabulary doesn't match.

### Query Rewriter
- Sends the question to GPT-4o-mini: *"rewrite this using technical terminology likely to appear in the source document"*
- Gets back one improved query
- That rewritten query goes to Qdrant instead of the original

### Multi-Query Retriever

**Full flow:**
```
4 queries (original + 3 GPT-4o-mini generated variants)
   ↓ fetch 5 chunks each
up to 20 chunks
   ↓ deduplicate by first 200 chars of content
N unique chunks (5–20)
   ↓ Flashrank reranker — scores all N chunks
top 3 chunks
   ↓ GPT-4o-mini
final answer
```

**How deduplication works:**
- Iterates queries in order: original → variant 1 → variant 2 → variant 3
- First time a chunk appears → kept
- Same chunk from a different query → dropped (first-seen wins)
- No scoring at this stage — that's the reranker's job

**Why multi-query helps:**
The reranker gets more diverse candidates to pick from, so the final top 3 are better quality.

### One-liner
> "Query rewriting translates natural language into paper-specific terminology before vector search. Multi-query generates 3 extra phrasings, retrieves 5 chunks per query, deduplicates, and gives the reranker a richer pool to pick the best 3 from."

---

## Reranker

- Flashrank (`ms-marco-MiniLM-L-12-v2`) scores every retrieved chunk
- Keeps top 3 (fallback to top 1 if everything scores below threshold)
- `rerank_threshold = 0.0` — best value found after sweeping 0.0, 0.1, 0.2, 0.3, 0.5. Any filtering dropped useful chunks.

---

## RAGAS Evaluation — How It Works

### What is RAGAS?
RAGAS is a framework that automatically evaluates a RAG pipeline using an LLM as a judge. Instead of manually checking answers, you give it questions, the pipeline's answers, the retrieved chunks, and ground-truth answers — it scores everything.

### What we evaluated on
- **Dataset**: 10 hand-crafted Q&A pairs about the "Attention Is All You Need" paper (`eval_data/golden_dataset.json`)
- **Fixed config**: StructureAwareChunker, chunk_size=1000, rerank_threshold=0.0
- **Variable**: 3 query strategies — none, rewrite, multi

### The 4 metrics and what they mean

| Metric | What it measures | Low score means |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved chunks? | The LLM is hallucinating — making up things not in the context |
| **Answer Relevancy** | Does the answer actually address the question? | The answer is off-topic or too vague |
| **Context Precision** | Are the retrieved chunks relevant? | Noisy retrieval — fetching irrelevant chunks |
| **Context Recall** | Do the chunks contain the info needed to answer? | Missing chunks — the right content wasn't retrieved at all |

### How the script works (`evaluate_rag.py`)
1. Loads the golden dataset (10 Q&A pairs)
2. For each strategy, runs all 10 questions through the full RAG pipeline
3. Collects: question, generated answer, retrieved chunks (`contexts`), ground truth
4. Passes everything to RAGAS → gets per-question scores
5. Averages scores across all 10 questions → aggregate scores
6. Saves results to `eval_results_struct_{strategy}.json`
7. Prints a comparison table at the end

### Results

| Metric | none | rewrite | multi |
|---|---|---|---|
| faithfulness | **0.9104** | NaN | 0.8917 |
| answer_relevancy | 0.8628 | 0.8726 | **0.8740** |
| context_precision | 0.6667 | 0.6667 | **0.6750** |
| context_recall | **0.8000** | 0.7500 | 0.7500 |

**Winner: "none"** — best faithfulness + recall, no NaN issues

---

## What I Learned

### 1. Chunking matters more than query strategy
Switching from fixed-size (500 chars) to structure-aware chunking improved context recall noticeably — concepts that were being split across chunk boundaries stayed together. Query strategies on the other hand barely moved the needle (precision 0.667 → 0.675).

### 2. The real bottleneck was retrieval quality, not generation
Faithfulness was 0.91 — the LLM was doing a great job answering from whatever chunks it got. But context precision was stuck at 0.667. The embedding model (`text-embedding-3-small`) couldn't distinguish between similar Transformer paper sections well. Query-side tricks can't fix an index-quality problem.

### 3. More filtering = worse results
Sweeping rerank threshold (0.0, 0.1, 0.2, 0.3, 0.5) showed that any score cutoff dropped useful chunks. Best to keep all top-3 and let the LLM figure it out.

### 4. NaN is a silent failure
The "rewrite" strategy returned NaN for faithfulness — RAGAS computed it without throwing an error, but the score was invalid. Learned to always sanity-check metric outputs, not just assume a number means correctness.

### 5. What I'd do next
Hybrid search (dense vectors + BM25 keyword search) to break the context precision ceiling — purely dense retrieval struggles with a domain where many sections use similar vocabulary.

---

## Good Follow-up Answers

| Question | Answer |
|---|---|
| "Why LangGraph?" | Gives explicit control over routing logic as a graph — easier to extend than a chain |
| "Why Qdrant?" | Fast cosine similarity search, easy Docker setup, native collection management |
| "What would you improve?" | Better embeddings or hybrid search (dense + BM25) to break the 0.667 context precision ceiling |
| "What did RAGAS tell you?" | Retrieval quality was the bottleneck, not generation — faithfulness was 0.91 but precision plateaued at 0.667 |
| "Why did multi-query not help much?" | The embedding model couldn't distinguish similar sections in the paper regardless of how we phrased the query |
