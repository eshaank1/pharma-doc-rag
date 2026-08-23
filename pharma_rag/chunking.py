from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from .schemas import ChunkMetadata, LogicalDocument


def chunk_document_with_metadata(logical_doc: LogicalDocument,
                                  chunk_size: int = 100,
                                  overlap: int = 20) -> List[ChunkMetadata]:
    """
    Chunk a logical document while preserving rich metadata.
    Uses a sliding window over whole LINES (not raw words split() and
    rejoined with spaces). The old word-based version flattened every
    newline -- including the one line-per-table-row structure produced by
    extract_page_text() -- into a single run of words, so cells from
    adjacent table rows became indistinguishable, and a chunk boundary
    could fall in the middle of that run and mix rows together with no
    way to tell which value belonged to which row.
    A line is only ever split mid-line if it alone exceeds chunk_size
    words (e.g. a long prose paragraph line rather than a table row), so
    a table row's cells always stay together in one chunk.
    """
    lines = [line for line in logical_doc.text.split("\n") if line.strip()]

    # Pre-split any line longer than chunk_size into word-sized pieces so
    # it can't block the sliding window from making progress.
    units = []
    for line in lines:
        words = line.split()
        if len(words) <= chunk_size:
            units.append(line)
        else:
            for start in range(0, len(words), chunk_size):
                units.append(' '.join(words[start:start + chunk_size]))

    if not units:
        return []

    def unit_words(u):
        return len(u.split())

    chunks_metadata = []
    n = len(units)
    i = 0
    chunk_index = 0

    while i < n:
        collected = []
        word_count = 0
        j = i
        while j < n and (word_count == 0 or word_count + unit_words(units[j]) <= chunk_size):
            collected.append(units[j])
            word_count += unit_words(units[j])
            j += 1

        chunk_text = "\n".join(collected)

        chunk_position = i / n
        page_range = logical_doc.page_end - logical_doc.page_start
        relative_page = int(chunk_position * page_range)
        chunk_page_start = logical_doc.page_start + relative_page
        chunk_page_end = min(chunk_page_start + 1, logical_doc.page_end)

        chunks_metadata.append(ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{chunk_index}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=chunk_index,
            page_start=chunk_page_start,
            page_end=chunk_page_end,
            text=chunk_text
        ))
        chunk_index += 1

        if j >= n:
            break

        # Step back by roughly `overlap` words worth of units so context
        # carries into the next chunk, same intent as the original
        # word-based stride -- but always measured in whole units/lines.
        back_words = 0
        k = j - 1
        while k > i and back_words < overlap:
            back_words += unit_words(units[k])
            k -= 1
        i = max(i + 1, k + 1)

    return chunks_metadata

    return chunks_metadata


def chunk_with_llama_index(logical_doc: LogicalDocument,
                            chunk_size: int = 100,
                            chunk_overlap: int = 20) -> List[ChunkMetadata]:
    """
    Alternative: Use LlamaIndex's advanced chunking with metadata.
    """
    doc = Document(
        text=logical_doc.text,
        metadata={
            "doc_id": logical_doc.doc_id,
            "doc_type": logical_doc.doc_type,
            "page_start": logical_doc.page_start,
            "page_end": logical_doc.page_end,
            "source": f"{logical_doc.doc_type}_document"
        }
    )

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        separator=" ",
    )

    nodes = splitter.get_nodes_from_documents([doc])

    chunks_metadata = []
    for i, node in enumerate(nodes):
        chunk_meta = ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            chunk_index=i,
            page_start=node.metadata.get("page_start", logical_doc.page_start),
            page_end=node.metadata.get("page_end", logical_doc.page_end),
            text=node.text
        )
        chunks_metadata.append(chunk_meta)

    return chunks_metadata


def process_all_documents(logical_docs: List[LogicalDocument],
                           use_llama_index: bool = False) -> List[ChunkMetadata]:
    """
    Process all logical documents into chunks with metadata.
    """
    all_chunks = []

    for logical_doc in logical_docs:
        if use_llama_index:
            chunks = chunk_with_llama_index(logical_doc)
        else:
            chunks = chunk_document_with_metadata(logical_doc)

        logical_doc.chunks = chunks
        all_chunks.extend(chunks)
        print(f"  {logical_doc.doc_type}: Created {len(chunks)} chunks")

    return all_chunks
