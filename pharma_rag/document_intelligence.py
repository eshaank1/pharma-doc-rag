import re

from .config import VALID_DOC_TYPES
from .llm import llm_generate

# Ordered (checked top to bottom, first match wins) title/keyword
# patterns matched against the START of a page's text.
TITLE_PATTERNS = [
    ("Cover Letter", re.compile(
        r"To Whom It May Concern|^\s*Dear\s|^\s*Re:\s|Sincerely,", re.I)),
    ("Certificate Of Quality", re.compile(
        r"Certificate of Quality", re.I)),
    ("Packaging Specification", re.compile(
        r"Packaging (Component )?Specification|PKG-SPEC", re.I)),
    ("BSE/TSE Declaration", re.compile(
        r"Transmissible Spongiform|BSE/TSE|Spongiform Encephalopath", re.I)),
    ("Material Description", re.compile(
        r"Material Description Sheet", re.I)),
    ("Supplier Qualification", re.compile(
        r"Supplier Qualification Record", re.I)),
    ("Chain Of Custody", re.compile(
        r"Chain of Custody", re.I)),
]

# Signals that a page CONTINUES the previous page rather than starting
# a new document (e.g. "(continued)", "Page 2 of 2").
CONTINUATION_PATTERN = re.compile(
    r"\(continued\)|continued\)|\bPage\s+(\d+)\s+of\s+(\d+)", re.I)


def _is_continuation_page(text: str, sample_len: int = 300) -> bool:
    """Heuristic: does this page explicitly mark itself as a continuation?"""
    sample = text[:sample_len]
    m = re.search(r"\bPage\s+(\d+)\s+of\s+(\d+)", sample, re.I)
    if m and int(m.group(1)) > 1:
        return True
    if re.search(r"\(continued\)", sample, re.I):
        return True
    return False


def classify_document_type_heuristic(text: str, sample_len: int = 400) -> str:
    """Fast, deterministic classification using known document headers."""
    sample = text[:sample_len]
    for doc_type, pattern in TITLE_PATTERNS:
        if pattern.search(sample):
            return doc_type
    return "Other"


def clean_doc_type(response):
    """Clean up LLM response to extract a valid doc_type label."""
    cleaned = response.strip().replace('"', '').replace('`', '').replace('*', '').lower().replace(".", "").strip()
    cleaned_title = cleaned.title()
    for label in VALID_DOC_TYPES:
        if label.lower() in cleaned.lower():
            return label
    return cleaned_title


def classify_document_type(text: str, max_length: int = 1500) -> str:
    """
    Classify the document type. Tries the fast heuristic first (reliable
    for structured pharma documents and has no external dependency);
    only falls back to the local LLM when the heuristic can't confidently
    label the page. LLM errors are logged instead of being silently
    swallowed.
    """
    heuristic_type = classify_document_type_heuristic(text)
    if heuristic_type != "Other":
        return heuristic_type

    text_sample = text[:max_length] if len(text) > max_length else text
    # NOTE: no manual [INST]/[/INST] markers here -- Ollama's model
    # template already wraps every prompt in them (confirmed via
    # `ollama show mistral --modelfile`), and [INST]/[/INST] are also
    # configured as stop sequences for this model. Adding them here too
    # produced a doubly-nested "[INST] [INST] ... [/INST][/INST]" prompt,
    # which degraded classification accuracy.
    prompt = f"""You are a pharmaceutical document classifier.
Classify the page into EXACTLY ONE of these types: {VALID_DOC_TYPES}

Definitions (check in this order):
- "Cover Letter": a LETTER. Look for "To Whom It May Concern", "Dear", "Re:",
  "Sincerely", or a signature block. This ALWAYS wins if the page is a letter,
  even if it mentions quality or part numbers.
- "Certificate of Quality": has Lot Number, Date of Manufacture, Expiration Date,
  and a test-results table (Autoclave, Gamma Irradiation, "Conforms").
- "Packaging Specification": packaging components, blister tray, lid film, carton,
  a PKG-SPEC document number.
- "BSE/TSE Declaration": declaration about animal-origin materials / TSE compliance.
- "Material Description": Materials of Construction table, sterilization compatibility,
  physical properties (dimensions, weight).
- "Supplier Qualification": Supplier Name/Code, audit history, ISO 9001/13485
  certifications, approved product list.
- "Chain of Custody": "Chain of Custody", list of assemblies, traceability flow.
- "Other": only if none clearly apply.

Page Content:
{text_sample[:2000]}

OUTPUT: Respond with ONLY the exact type name from the list."""
    try:
        return clean_doc_type(llm_generate(prompt))
    except Exception as e:
        import traceback
        print(f"Classification error (LLM unavailable, keeping heuristic result 'Other'): {e}")
        traceback.print_exc()
        return "Other"


def detect_document_boundary(prev_text: str, curr_text: str,
                              current_doc_type: str = None) -> bool:
    """
    Detect if two consecutive pages belong to the same document.
    Returns True if SAME document, False if a NEW document starts here.

    Heuristic-first:
      1. If the current page explicitly marks itself as a continuation
         ("(continued)", "Page 2 of 2") -> same document.
      2. Else if the current page opens with a recognizable document
         title/header -> new document (even if the title matches the
         previous type, e.g. two back-to-back single-page Certificates
         of Quality for different lots are still two separate documents).
      3. Otherwise, fall back to the LLM for a judgment call; if the LLM
         is unavailable, default to "same document" only as a last resort
         and log the real error.
    """
    if not prev_text or not curr_text:
        return False

    if _is_continuation_page(curr_text):
        return True

    curr_heuristic_type = classify_document_type_heuristic(curr_text)
    if curr_heuristic_type != "Other":
        return False  # fresh, recognizable header -> new document

    prev_sample = prev_text[-500:] if len(prev_text) > 500 else prev_text
    curr_sample = curr_text[:500] if len(curr_text) > 500 else curr_text
    prompt = f"""Determine if these two pages are from the SAME pharmaceutical document.
Current document type: {current_doc_type or 'Unknown'}

A NEW document starts when the page has:
- A different document title or heading (e.g., "Certificate of Quality"
  vs "Packaging Specification" vs "Material Description Sheet")
- A completely different topic or subject matter
- Its own header with a new document number or reference

Pages belong to the SAME document when:
- The second page says "continued" or "page 2 of 2"
- The content directly continues the previous page's discussion
- They share the same document number or title

End of Previous Page:
...{prev_sample}

Start of Current Page:
{curr_sample}...

Answer ONLY 'Yes' if same document or 'No' if different document."""
    try:
        return llm_generate(prompt).lower().startswith('yes')
    except Exception as e:
        import traceback
        print(f"Boundary detection error (LLM unavailable, defaulting to 'same document'): {e}")
        traceback.print_exc()
        return True
