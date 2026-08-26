from typing import Dict, List, Tuple

from .llm import llm_generate
from .schemas import ChunkMetadata


def generate_answer_with_sources(query: str,
                                  retrieved_chunks: List[Tuple[ChunkMetadata, float]]) -> Dict:
    """Generate answer with detailed source attribution using the local LLM."""
    if not retrieved_chunks:
        return {
            'answer': "I couldn't find relevant information to answer your question.",
            'sources': [],
            'confidence': 0.0
        }

    # De-duplicate overlapping lines per source document: the sliding-window
    # chunker repeats rows across adjacent chunks, which over-represents them
    # to the LLM and skews which facts a "concise" answer picks.
    groups = {}
    group_order = []
    for chunk_meta, score in retrieved_chunks:
        key = chunk_meta.doc_id
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append((chunk_meta, score))

    context_parts = []
    sources = []
    for key in group_order:
        group = sorted(groups[key], key=lambda cs: cs[0].chunk_index)
        seen_lines = set()
        merged_lines = []
        for chunk_meta, _ in group:
            for line in chunk_meta.text.split("\n"):
                if line not in seen_lines:
                    seen_lines.add(line)
                    merged_lines.append(line)
        merged_text = "\n".join(merged_lines)

        doc_type = group[0][0].doc_type
        page_start = min(c.page_start for c, _ in group)
        page_end = max(c.page_end for c, _ in group)
        avg_group_score = sum(s for _, s in group) / len(group)

        context_parts.append(f"[From {doc_type}, Pages {page_start}-{page_end}]")
        context_parts.append(merged_text)
        context_parts.append("")
        sources.append({
            'doc_type': doc_type,
            'pages': f"{page_start}-{page_end}",
            'relevance': f"{avg_group_score:.2%}",
            'preview': merged_text[:100] + "...",
            'text': merged_text,
        })
    context = "\n".join(context_parts)

    prompt = f"""You are answering questions about pharmaceutical documentation
including certificates of quality, packaging specifications, and compliance
declarations. Use the provided context to answer the question accurately.
Be specific and cite which document type and pages support your answer.

Context:
{context}

Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Mention which document type(s) contain the information
3. Be concise but complete
4. If the context doesn't contain enough information, say so

Answer:"""
    try:
        answer = llm_generate(prompt)
        avg_score = sum(s for _, s in retrieved_chunks) / len(retrieved_chunks)
        return {
            'answer': answer,
            'sources': sources,
            'confidence': avg_score,
            'chunks_used': len(retrieved_chunks)
        }
    except Exception as e:
        import traceback
        print(f"Answer generation error: {type(e).__name__}: {e!r}")
        traceback.print_exc()

        fallback_snippets = "\n\n".join(
            f"[{s['doc_type']}, Pages {s['pages']}]\n{s['text']}"
            for s in sources
        )
        return {
            'answer': (
                f"The AI answer-generation step failed "
                f"({type(e).__name__}: {str(e) or 'no error message -- see console/traceback for details'}). "
                f"Here are the relevant passages retrieved for your question instead:\n\n"
                f"{fallback_snippets}"
            ),
            'sources': sources,
            'confidence': 0.0
        }
