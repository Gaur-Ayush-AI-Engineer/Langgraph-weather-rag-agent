"""
Query strategies for improving retrieval precision.

Two strategies are provided:
  - QueryRewriter     : rewrites the original question once using an LLM so it
                        better matches terminology in the source document.
  - MultiQueryRetriever: generates N alternative phrasings of the question,
                        retrieves docs for each, and returns a deduplicated pool
                        for the reranker to work on.

Usage in rag.py:
    from query_strategies import QueryRewriter, MultiQueryRetriever

    rag = RAGTool(query_strategy=QueryRewriter())
    rag = RAGTool(query_strategy=MultiQueryRetriever(n_variants=3))
    rag = RAGTool()   # no strategy — original behaviour
"""

from __future__ import annotations

from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── Shared LLM (lightweight, deterministic) ────────────────────────────────────
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Query Rewriter ─────────────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at improving search queries for academic paper retrieval. "
        "Rewrite the user's question so it uses precise technical terminology that is "
        "likely to appear verbatim in the source document. "
        "Return ONLY the rewritten question — no explanation, no numbering.",
    ),
    ("human", "{question}"),
])


class QueryRewriter:
    """
    Rewrites a single question into a more document-friendly form before retrieval.

    The rewritten query replaces the original for the Qdrant embedding search.
    Everything downstream (reranker, LLM answer generation) is unchanged.
    """

    def __init__(self):
        self._chain = _REWRITE_PROMPT | _llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        """
        Args:
            question: Original user question.

        Returns:
            Rewritten query string.
        """
        rewritten = self._chain.invoke({"question": question}).strip()
        print(f"\n✏️  Query Rewriter")
        print(f"   Original : {question}")
        print(f"   Rewritten: {rewritten}")
        return rewritten


# ── Multi-Query Retriever ──────────────────────────────────────────────────────

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at generating search query variants for academic paper retrieval. "
        "Given a question, generate {n} different versions of it. Each version should "
        "emphasise a different aspect or use different terminology that might appear in the paper. "
        "Return ONLY the questions, one per line, with no numbering, bullets, or extra text.",
    ),
    ("human", "{question}"),
])


class MultiQueryRetriever:
    """
    Generates N alternative phrasings of the question, retrieves documents for
    each variant, and returns a deduplicated union for the reranker.

    Because the reranker receives more diverse candidates, it has a better chance
    of selecting the truly relevant chunks — improving context precision.
    """

    def __init__(self, n_variants: int = 3):
        self.n_variants = n_variants
        self._chain = _MULTI_QUERY_PROMPT | _llm | StrOutputParser()

    def expand(self, question: str) -> List[str]:
        """
        Generate query variants including the original question.

        Args:
            question: Original user question.

        Returns:
            List of query strings (original + variants).
        """
        raw = self._chain.invoke({"question": question, "n": self.n_variants})
        variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]

        # Always include the original so we never lose the exact phrasing
        all_queries = [question] + [v for v in variants if v != question]

        print(f"\n🔀 Multi-Query Expansion")
        print(f"   Original : {question}")
        for i, q in enumerate(all_queries[1:], 1):
            print(f"   Variant {i}: {q}")

        return all_queries

    def retrieve_and_deduplicate(self, question: str, retriever, k: int = 5):
        """
        Run retrieval for every query variant and return a deduplicated doc list.

        Args:
            question  : Original user question.
            retriever : LangChain retriever (e.g. vectorstore.as_retriever(...)).
            k         : Docs to fetch per query variant.

        Returns:
            Deduplicated list of LangChain Document objects.
        """
        queries = self.expand(question)
        seen_contents = set()
        unique_docs = []

        for q in queries:
            docs = retriever.invoke(q)
            for doc in docs:
                # Deduplicate by first 200 chars of content
                fingerprint = doc.page_content[:200]
                if fingerprint not in seen_contents:
                    seen_contents.add(fingerprint)
                    unique_docs.append(doc)

        print(f"   Retrieved {len(unique_docs)} unique docs across {len(queries)} queries")
        return unique_docs
