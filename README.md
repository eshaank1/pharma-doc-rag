# pharma-doc-rag
RAG pipeline that splits multi-document pharmaceutical PDFs into logical sections, indexes them, and auto-routes questions to the right document type via a Gradio chat UI.

Originally prototyped as a Colab notebook (`FullProject.ipynb`) using Mistral-7B via `bitsandbytes` 4-bit quantization, which requires a CUDA GPU. This version runs entirely locally on any machine by serving the LLM through [Ollama](https://ollama.com) instead.

## Setup

Requires **Python 3.10+** (`llama-index-core`'s dependencies use syntax that doesn't run on 3.9).

1. **Install Ollama** and pull a model:
   ```bash
   brew install ollama
   ollama serve                 # run this in a separate terminal, keep it running
   ollama pull mistral          # or: ollama pull qwen2.5:7b-instruct
   ```
   Set `OLLAMA_MODEL` (env var) if you pull a different model than `mistral`.

2. **Install Tesseract OCR** (needed for scanned/image-only PDF pages):
   ```bash
   brew install tesseract
   ```

3. **Create a virtualenv and install Python dependencies:**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Running it

```bash
./start.sh
```
Starts Ollama (if it isn't already running) and launches the app at **http://127.0.0.1:7860**. Upload a pharmaceutical blob PDF and start asking questions.

```bash
./stop.sh
```
Stops both the app and Ollama.

(A Docker-based setup for distributing this to others is planned but not included yet.)

## Configuration

Environment variables (all optional, see `pharma_rag/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | Model to use for classification, routing, and answer generation |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |

## Project layout

```
start.sh                         # starts Ollama (if needed) and the app
stop.sh                          # stops the app and Ollama
main.py                          # entry point: launches the Gradio app
pharma_rag/
  config.py                      # constants and env-var configuration
  llm.py                         # Ollama-backed llm_generate() + JSON extraction
  schemas.py                     # PageInfo / LogicalDocument / ChunkMetadata dataclasses
  document_intelligence.py       # heuristic + LLM document classification and boundary detection
  pdf_processing.py              # PDF text extraction, OCR pipeline, text cleaning
  chunking.py                    # sliding-window and LlamaIndex-based chunking
  retrieval.py                   # embeddings, FAISS indices, query routing
  answer_generation.py           # source-attributed answer generation
  document_store.py              # EnhancedDocumentStore: ties the pipeline together
  ui.py                          # Gradio Blocks interface, theme, and CSS
```
