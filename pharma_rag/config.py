import os

# ---- Local LLM (served by Ollama, e.g. `ollama serve` + `ollama pull mistral`) ----
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

# Ollama sizes its runtime context window heuristically (often just 4096
# tokens) rather than using the model's full supported context, unless
# told otherwise. summarize_all_documents() concatenates the full text of
# every logical document into one prompt, which can easily exceed 4096
# tokens for a multi-document pharma PDF -- past that, Ollama silently
# drops/shifts out earlier context instead of erroring, producing answers
# based on a truncated document. Set explicitly, well above the default.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))

# ---- Embedding model (shared by the retriever and LlamaIndex-based chunking) ----
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

VALID_DOC_TYPES = [
    "Cover Letter", "Certificate Of Quality", "Packaging Specification",
    "BSE/TSE Declaration", "Material Description", "Supplier Qualification",
    "Chain Of Custody", "Other"
]

# Below this length, page text is treated as too unreliable to classify
# or boundary-check with any confidence (empty extraction, failed OCR,
# or a near-blank page).
MIN_TEXT_LENGTH = 20

# Below this mean Tesseract word-confidence (0-100), an OCR'd page is
# flagged as low-confidence rather than trusted outright.
MIN_OCR_CONFIDENCE = 60
