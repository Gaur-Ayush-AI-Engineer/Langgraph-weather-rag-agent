"""
RAGAS evaluation script for the Attention Is All You Need RAG pipeline.

Run: python evaluate_rag.py
Outputs: prints a results table + saves eval_results.json
"""

import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()

PDF_PATH            = "./eval_data/attention_is_all_you_need.pdf"
GOLDEN_DATASET_PATH = "./eval_data/golden_dataset.json"

# ── Experiment config ──────────────────────────────────────────────────────────
USE_STRUCTURE_AWARE_CHUNKING = True

CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
RERANK_THRESHOLD = 0.0   # use best threshold from previous sweep

# Query strategies to compare — each gets its own evaluation run and results file
# Options: "none" | "rewrite" | "multi"
STRATEGIES = ["none", "rewrite", "multi"]

# ── Load Golden Dataset ────────────────────────────────────────────────────────
with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as _f:
    GOLDEN_DATASET = json.load(_f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_ragas_dataset(results):
    """
    Build a RAGAS-compatible dataset from the pipeline results.
    Supports both ragas 0.1.x (datasets.Dataset) and ragas 0.2.x (EvaluationDataset).
    """
    questions   = [r["question"]    for r in results]
    answers     = [r["answer"]      for r in results]
    contexts    = [r["contexts"]    for r in results]
    references  = [r["ground_truth"] for r in results]

    # Try ragas 0.2.x API first
    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample

        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=c,
                reference=g,
            )
            for q, a, c, g in zip(questions, answers, contexts, references)
        ]
        return EvaluationDataset(samples=samples), "v2"
    except ImportError:
        pass

    # Fall back to ragas 0.1.x API (datasets.Dataset)
    from datasets import Dataset

    data = {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": references,
    }
    return Dataset.from_dict(data), "v1"


def get_metrics(api_version):
    """
    Return metric objects with explicitly configured LLM + embeddings wrappers.
    RAGAS requires its own LangchainLLMWrapper / LangchainEmbeddingsWrapper —
    passing raw LangChain objects causes AttributeError on embed_query.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=8800))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    if api_version == "v2":
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        return [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ]
    else:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings
        context_precision.llm = ragas_llm
        context_recall.llm = ragas_llm
        return [faithfulness, answer_relevancy, context_precision, context_recall]


def extract_scores(per_question):
    """Compute aggregate scores by averaging per-question scores."""
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {}
    for k in keys:
        values = [r[k] for r in per_question if r.get(k) is not None]
        scores[k] = round(sum(values) / len(values), 4) if values else None
    return scores


def extract_per_question(eval_result, results):
    """Merge per-question RAGAS scores with the original pipeline results."""
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    try:
        df = eval_result.to_pandas()
        rows = df.to_dict(orient="records")
    except Exception:
        rows = [{} for _ in results]

    combined = []
    for i, r in enumerate(results):
        row = rows[i] if i < len(rows) else {}
        combined.append({
            "question":          r["question"],
            "ground_truth":      r["ground_truth"],
            "answer":            r["answer"],
            "contexts":          r["contexts"],
            "faithfulness":      float(row.get("faithfulness",      row.get("Faithfulness",      None) or 0)),
            "answer_relevancy":  float(row.get("answer_relevancy",  row.get("AnswerRelevancy",   None) or 0)),
            "context_precision": float(row.get("context_precision", row.get("ContextPrecision",  None) or 0)),
            "context_recall":    float(row.get("context_recall",    row.get("ContextRecall",     None) or 0)),
        })
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────

def run_single(rag, api_version, threshold_label):
    """Run the full RAG + RAGAS pipeline for one threshold setting."""
    from ragas import evaluate

    # Run each question
    print(f"\n🔍 Running {len(GOLDEN_DATASET)} questions (threshold={threshold_label})...")
    pipeline_results = []
    for i, item in enumerate(GOLDEN_DATASET, 1):
        print(f"  [{i:02d}/{len(GOLDEN_DATASET)}] {item['question'][:70]}...")
        result = rag.query(item["question"])
        pipeline_results.append({
            "question":     item["question"],
            "ground_truth": item["ground_truth"],
            "answer":       result["answer"],
            "contexts":     result["contexts"],
        })

    # Build dataset and evaluate
    dataset, _ = build_ragas_dataset(pipeline_results)
    metrics = get_metrics(api_version)
    print("⏳ Running RAGAS evaluation...\n")
    eval_result = evaluate(dataset, metrics=metrics)

    per_question = extract_per_question(eval_result, pipeline_results)
    scores = extract_scores(per_question)
    return scores, per_question


def run_evaluation():
    # ── 1. Validate PDF ────────────────────────────────────────────────────────
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at {PDF_PATH}")
        print("   Please place 'attention_is_all_you_need.pdf' in the eval_data/ directory.")
        sys.exit(1)

    from rag import RAGTool

    # ── 2. Setup chunker and ingest PDF once ───────────────────────────────────
    if USE_STRUCTURE_AWARE_CHUNKING:
        from chunkers import StructureAwareChunker
        chunker = StructureAwareChunker(max_chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"🚀 Using StructureAwareChunker (max_chunk_size={CHUNK_SIZE})")
    else:
        chunker = None
        print(f"🚀 Using RecursiveCharacterTextSplitter (chunk_size={CHUNK_SIZE})")

    # Ingest once with threshold=0.0 (threshold only affects retrieval, not ingestion)
    base_rag = RAGTool(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, chunker=chunker, rerank_threshold=0.0)
    if not base_rag.load_pdf(PDF_PATH):
        print("❌ Failed to load PDF. Is Qdrant running on port 6333?")
        sys.exit(1)

    # Detect api_version once
    _, api_version = build_ragas_dataset([{
        "question": "x", "ground_truth": "x", "answer": "x", "contexts": ["x"]
    }])

    # ── 3. Sweep query strategies ──────────────────────────────────────────────
    from query_strategies import QueryRewriter, MultiQueryRetriever

    strategy_objects = {
        "none":    None,
        "rewrite": QueryRewriter(),
        "multi":   MultiQueryRetriever(n_variants=3),
    }

    chunker_tag = "struct" if USE_STRUCTURE_AWARE_CHUNKING else f"chunk{CHUNK_SIZE}"
    all_results = {}

    for strategy_name in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"  QUERY STRATEGY = {strategy_name}")
        print(f"{'='*60}")

        rag = RAGTool(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            chunker=chunker,
            rerank_threshold=RERANK_THRESHOLD,
            query_strategy=strategy_objects[strategy_name],
        )
        rag.load_pdf(PDF_PATH)

        scores, per_question = run_single(rag, api_version, strategy_name)
        all_results[strategy_name] = scores

        results_path = f"eval_results_{chunker_tag}_{strategy_name}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"strategy": strategy_name, "aggregate_scores": scores, "per_question": per_question},
                      f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Saved → {results_path}")

    # ── 4. Final comparison table ──────────────────────────────────────────────
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    col_w = 14

    print("\n\n" + "=" * (22 + col_w * len(STRATEGIES)))
    print("  QUERY STRATEGY COMPARISON")
    print("=" * (22 + col_w * len(STRATEGIES)))
    header = f"  {'Metric':<20}" + "".join(f"  {s:<{col_w-2}}" for s in STRATEGIES)
    print(header)
    print("-" * (22 + col_w * len(STRATEGIES)))
    for metric in metrics:
        row = f"  {metric:<20}"
        for s in STRATEGIES:
            val = all_results[s].get(metric)
            row += f"  {val:.4f}      " if val is not None else f"  {'N/A':<{col_w-2}}"
        print(row)
    print("=" * (22 + col_w * len(STRATEGIES)))

    avg_scores = {
        s: sum(v for v in sc.values() if v is not None) / max(1, sum(1 for v in sc.values() if v is not None))
        for s, sc in all_results.items()
    }
    best = max(avg_scores, key=avg_scores.get)
    print(f"\n🏆 Best strategy: '{best}'  (avg score: {avg_scores[best]:.4f})")
    print("\n✅ Evaluation complete.\n")


if __name__ == "__main__":
    run_evaluation()
