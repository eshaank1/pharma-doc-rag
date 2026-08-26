from typing import List, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL_NAME
from .document_intelligence import clean_doc_type
from .llm import extract_json, llm_generate
from .schemas import ChunkMetadata

embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def predict_query_document_type(query: str) -> Tuple[str, float]:
    """Predict which pharmaceutical document type likely holds the answer."""
    prompt = f"""Analyze this query and predict which pharmaceutical document type
would most likely contain the answer.

Query: "{query}"

Choose the MOST LIKELY type from:
- Cover Letter: Formal letters about product information or storage conditions
- Certificate Of Quality: Lot numbers, manufacture/expiration dates, test results
- Packaging Specification: Packaging components, materials, part numbers
- BSE/TSE Declaration: Animal-origin material declarations, TSE compliance
- Material Description: Materials of construction, sterilization compatibility
- Supplier Qualification: Supplier audits, ISO certifications, approved products
- Chain Of Custody: Manufactured assemblies, traceability, shipment flow
- Other: General or unclear queries

Respond in JSON format:
{{"type": "DocumentType", "confidence": 0.85}}
Confidence should be between 0.0 and 1.0"""
    try:
        result = extract_json(llm_generate(prompt))
        predicted = result.get("type", "Other")
        confidence = float(result.get("confidence", 0.5))
        return clean_doc_type(predicted), confidence
    except Exception as e:
        print(f"Query routing error: {e}")
        return "Other", 0.0


class IntelligentRetriever:
    """
    Advanced retrieval system with metadata filtering and query routing.
    """

    def __init__(self):
        self.index = None
        self.chunks_metadata = []
        self.doc_type_indices = {}

    def build_indices(self, chunks_metadata: List[ChunkMetadata]):
        """
        Build FAISS indices with document type segregation.
        """
        print("Building vector indices...")
        self.chunks_metadata = chunks_metadata

        texts = [chunk.text for chunk in chunks_metadata]
        embeddings = embed_model.encode(texts, show_progress_bar=True)

        for i, chunk in enumerate(chunks_metadata):
            chunk.embedding = embeddings[i]

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        doc_types = set(chunk.doc_type for chunk in chunks_metadata)
        for doc_type in doc_types:
            type_indices = [i for i, chunk in enumerate(chunks_metadata)
                             if chunk.doc_type == doc_type]
            if type_indices:
                type_embeddings = embeddings[type_indices]
                type_index = faiss.IndexFlatL2(dim)
                type_index.add(type_embeddings)
                self.doc_type_indices[doc_type] = {
                    'index': type_index,
                    'mapping': type_indices
                }

        print(f"Indexed {len(chunks_metadata)} chunks across {len(doc_types)} document types")

    def retrieve(self, query: str, k: int = 4,
                 filter_doc_type: Optional[str] = None,
                 auto_route: bool = True) -> List[Tuple[ChunkMetadata, float]]:
        """
        Retrieve relevant chunks with optional filtering and routing.
        Returns chunks with relevance scores.
        """
        query_embedding = embed_model.encode([query])

        def _valid_hits(D, I):
            # FAISS pads short results with index -1; without filtering,
            # Python's negative indexing would turn that into "last item"
            # instead of "no match", duplicating a chunk into the results.
            return [(i, d) for i, d in zip(I[0], D[0]) if i != -1]

        if filter_doc_type and filter_doc_type in self.doc_type_indices:
            type_data = self.doc_type_indices[filter_doc_type]
            D, I = type_data['index'].search(query_embedding, k)
            hits = _valid_hits(D, I)
            chunk_indices = [type_data['mapping'][i] for i, _ in hits]
            distances = [d for _, d in hits]
        elif auto_route:
            predicted_type, confidence = predict_query_document_type(query)
            print(f"Query routed to: {predicted_type} (confidence: {confidence:.2f})")

            if confidence > 0.7 and predicted_type in self.doc_type_indices:
                type_data = self.doc_type_indices[predicted_type]
                D, I = type_data['index'].search(query_embedding, k)
                hits = _valid_hits(D, I)
                chunk_indices = [type_data['mapping'][i] for i, _ in hits]
                distances = [d for _, d in hits]
            else:
                D, I = self.index.search(query_embedding, k)
                hits = _valid_hits(D, I)
                chunk_indices = [i for i, _ in hits]
                distances = [d for _, d in hits]
        else:
            D, I = self.index.search(query_embedding, k)
            hits = _valid_hits(D, I)
            chunk_indices = [i for i, _ in hits]
            distances = [d for _, d in hits]

        scores = [max(0.0, 1 - (d / 2)) for d in distances]

        results = [(self.chunks_metadata[i], scores[idx])
                   for idx, i in enumerate(chunk_indices)]

        return results
