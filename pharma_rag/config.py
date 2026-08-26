import os

# ---- Local LLM (served by Ollama, e.g. `ollama serve` + `ollama pull mistral`) ----
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

# Ollama defaults to a small context window (often 4096 tokens) and silently
# truncates instead of erroring past it, which can cut off multi-document
# prompts from summarize_all_documents(). Set explicitly, well above default.
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
