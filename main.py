import os

from pharma_rag.llm import ensure_model_pulled, llm_self_test, wait_for_ollama
from pharma_rag.pdf_processing import check_tesseract
from pharma_rag.ui import COMPACT_CSS, LAB_THEME, create_interface

if __name__ == "__main__":
    check_tesseract()

    if wait_for_ollama():
        ensure_model_pulled()
    else:
        print("WARNING: could not reach the Ollama server; LLM calls will fail "
              "until it's reachable.")

    llm_self_test()

    demo = create_interface()
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(theme=LAB_THEME, css=COMPACT_CSS, server_name=server_name)
