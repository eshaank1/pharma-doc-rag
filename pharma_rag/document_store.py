from datetime import datetime
from typing import Dict, List, Optional

from .answer_generation import generate_answer_with_sources
from .chunking import process_all_documents
from .llm import llm_generate
from .pdf_processing import extract_and_analyze_pdf
from .retrieval import IntelligentRetriever


class EnhancedDocumentStore:
    """
    Manages the complete document processing and retrieval pipeline.
    """

    def __init__(self):
        self.pages_info = []
        self.logical_docs = []
        self.chunks_metadata = []
        self.retriever = IntelligentRetriever()
        self.is_ready = False
        self.processing_stats = {}
        self.filename = None

    def process_pdf(self, pdf_file, filename: str = "document.pdf"):
        """
        Complete PDF processing pipeline.
        """
        self.filename = filename
        self.is_ready = False
        start_time = datetime.now()

        try:
            self.pages_info, self.logical_docs = extract_and_analyze_pdf(pdf_file)
            self.chunks_metadata = process_all_documents(self.logical_docs)
            self.retriever.build_indices(self.chunks_metadata)

            process_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats = {
                'filename': filename,
                'total_pages': len(self.pages_info),
                'documents_found': len(self.logical_docs),
                'total_chunks': len(self.chunks_metadata),
                'document_types': list(set(doc.doc_type for doc in self.logical_docs)),
                'processing_time': f"{process_time:.1f}s"
            }

            self.is_ready = True
            return True, self.processing_stats

        except Exception as e:
            return False, {'error': str(e)}

    def query(self, question: str, filter_type: Optional[str] = None,
              auto_route: bool = True, k: int = 4) -> Dict:
        """
        Query the document store.
        """
        if not self.is_ready:
            return {
                'answer': "Please upload and process a PDF first.",
                'sources': [],
                'confidence': 0.0
            }

        retrieved = self.retriever.retrieve(
            question, k=k,
            filter_doc_type=filter_type,
            auto_route=auto_route
        )

        result = generate_answer_with_sources(question, retrieved)
        result['filter_used'] = filter_type or ('auto' if auto_route else 'none')

        return result

    def summarize_all_documents(self) -> Dict:
        """
        Summarize every identified document, not just the top-k most
        similar to a generic "summarize" query.
        """
        if not self.is_ready:
            return {
                'answer': "Please upload and process a PDF first.",
                'sources': [],
                'confidence': 0.0
            }

        if not self.logical_docs:
            return {
                'answer': "No documents were identified in this PDF.",
                'sources': [],
                'confidence': 0.0
            }

        context_parts = []
        sources = []
        for doc in self.logical_docs:
            context_parts.append(
                f"[{doc.doc_type}, Pages {doc.page_start + 1}-{doc.page_end + 1}]"
            )
            context_parts.append(doc.text)
            context_parts.append("")
            sources.append({
                'doc_type': doc.doc_type,
                'pages': f"{doc.page_start + 1}-{doc.page_end + 1}",
                'relevance': "100.00%",
                'preview': doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
            })
        context = "\n".join(context_parts)

        prompt = f"""You are summarizing a set of pharmaceutical documents.
Below is the FULL text of every document found in this PDF ({len(self.logical_docs)}
documents total). Write a clear summary that covers EVERY document listed --
do not skip any of them.

{context}

Instructions:
1. Summarize each of the {len(self.logical_docs)} documents listed above, in order
2. For each, state its document type and the key facts (lot numbers, part
   numbers, dates, key findings, etc. as applicable)
3. Be concise per document, but make sure all {len(self.logical_docs)} are covered

Summary:"""
        try:
            answer = llm_generate(prompt, max_new_tokens=1024)
            return {
                'answer': answer,
                'sources': sources,
                'confidence': 1.0,
                'chunks_used': len(self.logical_docs)
            }
        except Exception as e:
            import traceback
            print(f"Full-document summary error: {type(e).__name__}: {e!r}")
            traceback.print_exc()
            fallback = "\n\n".join(
                f"[{s['doc_type']}, Pages {s['pages']}]\n{doc.text}"
                for doc, s in zip(self.logical_docs, sources)
            )
            return {
                'answer': (
                    f"The AI summary step failed "
                    f"({type(e).__name__}: {str(e) or 'no error message -- see console/traceback for details'}). "
                    f"Here is the full text of all {len(self.logical_docs)} documents instead:\n\n{fallback}"
                ),
                'sources': sources,
                'confidence': 0.0
            }

    def get_document_structure(self) -> List[Dict]:
        """
        Get the document structure for UI display.
        """
        if not self.logical_docs:
            return []

        structure = []
        for doc in self.logical_docs:
            structure.append({
                'id': doc.doc_id,
                'type': doc.doc_type,
                'pages': f"{doc.page_start + 1}-{doc.page_end + 1}",  # 1-indexed for UI
                'chunks': len(doc.chunks) if doc.chunks else 0,
                'preview': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            })

        return structure
