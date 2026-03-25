"""
Structure-aware chunking for academic PDFs.

Splits documents by detected section headers first, then sub-splits
large sections at paragraph boundaries — never at fixed character counts.
Each chunk carries section metadata for richer retrieval context.
"""

import re
from typing import List, Optional
from langchain_core.documents import Document


# Matches numbered section headers common in academic papers:
# e.g. "1 Introduction", "3.1 Encoder and Decoder Stacks", "6.1 Machine Translation"
SECTION_HEADER_RE = re.compile(
    r'(?m)^(\d+(?:\.\d+)*)\s{1,4}([A-Z][A-Za-z0-9 \-,:]{2,60})$'
)


class StructureAwareChunker:
    """
    Chunks PDF documents by structural boundaries rather than fixed character counts.

    Strategy:
      1. Concatenate all page text from the loaded documents.
      2. Detect section headers using a regex pattern for numbered headings.
      3. Split the full text into per-section blocks.
      4. For sections that exceed max_chunk_size, sub-split at paragraph
         boundaries (double newlines), never mid-sentence.
      5. Prepend the section title to every sub-chunk so retrieval context
         always knows which section a passage belongs to.

    Usage:
        chunker = StructureAwareChunker(max_chunk_size=1000, chunk_overlap=100)
        splits = chunker.split_documents(documents)   # same interface as LangChain splitters
    """

    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public interface ───────────────────────────────────────────────────────

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of LangChain Documents into structure-aware chunks.

        Args:
            documents: Raw pages as returned by PyPDFLoader.

        Returns:
            List of Document objects, each with enriched metadata:
              - source, page (from original doc)
              - section_number, section_title
              - chunk_index (position within the section)
        """
        full_text = "\n".join(doc.page_content for doc in documents)
        base_metadata = documents[0].metadata if documents else {}

        sections = self._split_into_sections(full_text)
        chunks: List[Document] = []

        for section in sections:
            section_chunks = self._chunk_section(section)
            for idx, text in enumerate(section_chunks):
                metadata = {
                    **base_metadata,
                    "section_number": section["number"],
                    "section_title":  section["title"],
                    "chunk_index":    idx,
                }
                chunks.append(Document(page_content=text, metadata=metadata))

        print(
            f"✂️  StructureAwareChunker: {len(sections)} sections → {len(chunks)} chunks"
            f" (max_chunk_size={self.max_chunk_size})"
        )
        return chunks

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _split_into_sections(self, text: str) -> List[dict]:
        """
        Detect section headers and split the full document text into sections.
        Returns a list of dicts: {number, title, content}.
        Any text before the first header is grouped as section "0 Preamble".
        """
        matches = list(SECTION_HEADER_RE.finditer(text))

        if not matches:
            # No headers found — treat the whole document as one section
            return [{"number": "0", "title": "Document", "content": text}]

        sections = []

        # Text before the first header
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append({"number": "0", "title": "Preamble", "content": preamble})

        for i, match in enumerate(matches):
            number = match.group(1)
            title  = match.group(2).strip()
            start  = match.end()
            end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            if content:
                sections.append({"number": number, "title": title, "content": content})

        return sections

    def _chunk_section(self, section: dict) -> List[str]:
        """
        Turn a single section into one or more text chunks.

        - If the section fits within max_chunk_size, return it as one chunk.
        - Otherwise split at paragraph boundaries and merge paragraphs greedily
          until the chunk would exceed max_chunk_size, then start a new chunk.
        - Every chunk is prefixed with "[Section {number}: {title}]" so the LLM
          always has section context even after splitting.
        """
        header  = f"[Section {section['number']}: {section['title']}]\n"
        content = section["content"]

        full_text = header + content
        if len(full_text) <= self.max_chunk_size:
            return [full_text]

        # Split into paragraphs (one or more blank lines)
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', content) if p.strip()]

        chunks: List[str] = []
        current_parts: List[str] = [header]
        current_len: int = len(header)

        for para in paragraphs:
            para_len = len(para) + 1  # +1 for the newline separator

            if current_len + para_len > self.max_chunk_size and len(current_parts) > 1:
                # Save current chunk
                chunks.append("\n".join(current_parts).strip())

                # Start new chunk with overlap: carry the last paragraph forward
                overlap_text = current_parts[-1] if self.chunk_overlap > 0 else ""
                current_parts = [header]
                current_len   = len(header)
                if overlap_text:
                    current_parts.append(overlap_text)
                    current_len += len(overlap_text) + 1

            current_parts.append(para)
            current_len += para_len

        # Flush remaining
        if len(current_parts) > 1:
            chunks.append("\n".join(current_parts).strip())

        return chunks if chunks else [full_text]
