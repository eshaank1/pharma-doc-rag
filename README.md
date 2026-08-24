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

## Testing & evaluation

Correctness here isn't just "does it run" -- for a pharma document pipeline, the questions that matter are "did it retrieve the right document" and "did the answer actually contain the right fact." This repo has two layers of testing:

**Unit tests** (`tests/eval_lib/`) are ordinary fast pytest tests for the scoring logic itself (answer matching, recall checks, bucket classification):
```bash
.venv/bin/python -m pytest tests/eval_lib/ -v
```

**The evaluation harness** (`tests/run_eval.py`) runs the *real* pipeline -- real Ollama LLM calls, no mocking -- against a set of realistic synthetic pharma documents (certificates of quality, packaging specs, BSE/TSE declarations, supplier records, both born-digital and scanned/OCR'd) checked into `tests/fixtures/`, alongside hand-written ground truth in `tests/ground_truth/` (expected facts, expected document type, expected source pages) for each question. For every case it runs the full `process_pdf` -> `query` flow and scores:

- **Answer Match %** -- do all expected facts appear in the generated answer
- **Retrieval Recall@k** -- did retrieval surface a chunk from the expected page range
- **Avg Latency** -- end-to-end query time
- **Error Rate** -- PDF-processing failures, unhandled exceptions, and detected LLM-generation failures

broken out separately for **digital** vs **scanned** documents, since OCR'd text is noisier and stresses the pipeline differently.

Snapshot from a recent run (15 cases across 6 fixtures, `mistral` via Ollama) -- numbers will shift slightly run to run since the LLM isn't deterministic:

| | Digital | Scanned | Overall |
|---|---|---|---|
| Answer Match % | 92% | 100% | 93% |
| Retrieval Recall@4 | 92% | 100% | 93% |
| Avg Latency (s) | 4.5 | 2.2 | 4.2 |
| Error Rate | 0% | 0% | 0% |

The one recurring miss is a case where the model answers from an adjacent-but-wrong fact (an operating temperature instead of a storage temperature) rather than a retrieval or infrastructure failure -- the per-case JSON output records enough detail (including whether the retrieved sources matched the expected document type) to distinguish that kind of miss from a doc-type misroute or a dropped source.

To run it yourself:
```bash
.venv/bin/python tests/run_eval.py     # requires Ollama running, same as the app
```
This makes real LLM calls against every fixture, so a full run takes several minutes -- it's an evaluation suite, not a fast unit-test suite. Results are written as a timestamped JSON file to `tests/results/` (gitignored -- it's run output, not source).

`tests/fixtures/` and `tests/ground_truth/` are already committed, so `tests/generate_fixtures.py` normally doesn't need to be re-run -- it exists so the fixtures can be regenerated or audited if their content ever needs to change.

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
