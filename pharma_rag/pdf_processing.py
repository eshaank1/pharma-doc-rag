import io
import re
import shutil
import unicodedata
from typing import List, Tuple

import cv2
import pymupdf as fitz
import numpy as np

from .config import MIN_OCR_CONFIDENCE, MIN_TEXT_LENGTH
from .document_intelligence import classify_document_type, detect_document_boundary
from .schemas import LogicalDocument, PageInfo

# Table-aware text extraction.
#
# page.get_text() puts each table cell on its own line with no row
# markers, so cells from different rows can blur together downstream.
# Fix: reconstruct each row as one line ("cell | cell | ...") by
# clustering words into visual lines by y-position, splitting into
# columns on unusually wide horizontal gaps, and requiring >=3
# consecutive same-column-count lines to call it a table (works for
# both ruled and borderless tables; more reliable in testing than
# PyMuPDF's own find_tables(), which merged a whole non-table page
# into one bogus table).

_TABLE_GAP_THRESHOLD = 18.0   # points; well beyond a normal inter-word gap
_TABLE_Y_TOLERANCE = 3.0      # points; words within this are "the same line"
_TABLE_MIN_ROWS = 3
_TABLE_COL_RANGE = (2, 6)


def _cluster_page_lines(page) -> List[Tuple[float, float, List[str]]]:
    """Group a page's words into visual lines by y-position (robust to
    text written as separate content-stream runs at the same row, which
    don't always share a PyMuPDF block/line number), then split each line
    into column-like clusters wherever the horizontal gap is unusually
    wide. Returns (y0, y1, column_texts) per line, top to bottom."""
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, word_no)
    if not words:
        return []

    words = sorted(words, key=lambda w: (round(w[1] / _TABLE_Y_TOLERANCE), w[0]))

    raw_lines = []
    current, current_y = [], None
    for w in words:
        y0 = w[1]
        if current_y is None or abs(y0 - current_y) <= _TABLE_Y_TOLERANCE:
            current.append(w)
            current_y = current_y if current_y is not None else y0
        else:
            raw_lines.append(current)
            current, current_y = [w], y0
    if current:
        raw_lines.append(current)

    lines = []
    for ws in raw_lines:
        ws = sorted(ws, key=lambda w: w[0])
        y0 = min(w[1] for w in ws)
        y1 = max(w[3] for w in ws)
        clusters = [[ws[0]]]
        for w in ws[1:]:
            prev_end = clusters[-1][-1][2]
            if w[0] - prev_end >= _TABLE_GAP_THRESHOLD:
                clusters.append([w])
            else:
                clusters[-1].append(w)
        col_texts = [' '.join(x[4] for x in c) for c in clusters]
        lines.append((y0, y1, col_texts))

    lines.sort(key=lambda l: l[0])
    return lines


def _detect_table_blocks(lines: List[Tuple[float, float, List[str]]]) -> List[Tuple[int, int]]:
    """Find runs of >=_TABLE_MIN_ROWS consecutive lines that all split
    into the same number of columns (2-6). Returns [start, end) index
    ranges into `lines`."""
    blocks = []
    i = 0
    while i < len(lines):
        col_count = len(lines[i][2])
        if col_count < _TABLE_COL_RANGE[0] or col_count > _TABLE_COL_RANGE[1]:
            i += 1
            continue
        j = i
        while j < len(lines) and len(lines[j][2]) == col_count:
            j += 1
        if j - i >= _TABLE_MIN_ROWS:
            blocks.append((i, j))
        i = j
    return blocks


def extract_page_text(page) -> str:
    """Extract a page's text with table rows reconstructed as one line
    per row (cells joined by " | ", wrapped in [TABLE]/[/TABLE] markers)
    instead of PyMuPDF's default one-line-per-cell output. Falls back to
    plain page.get_text() when no table-like block is detected."""
    lines = _cluster_page_lines(page)
    if not lines:
        return page.get_text()

    blocks = _detect_table_blocks(lines)
    in_block_end = {}  # start index -> end index, for quick lookup
    in_block = [False] * len(lines)
    for start, end in blocks:
        in_block_end[start] = end
        for k in range(start, end):
            in_block[k] = True

    parts = []
    i = 0
    while i < len(lines):
        if i in in_block_end:
            end = in_block_end[i]
            row_texts = [' | '.join(lines[k][2]) for k in range(i, end)]
            parts.append("[TABLE]\n" + "\n".join(row_texts) + "\n[/TABLE]")
            i = end
        else:
            parts.append(' '.join(lines[i][2]))
            i += 1

    return "\n".join(parts)


def check_tesseract() -> None:
    """Tesseract failing silently is hard to diagnose: pytesseract just
    swallows the error and every scanned page comes back as "". Check
    once, up front, and say so loudly instead of leaving it as a mystery."""
    path = shutil.which("tesseract")
    if path is None:
        print("WARNING: Tesseract OCR binary not found on PATH.")
        print("  Scanned pages will silently extract as empty text.")
        print("  Install it with: brew install tesseract")
    else:
        print(f"Tesseract OCR found: {path}")


def clean_extracted_text(text: str) -> str:
    """General-purpose cleanup applied to every page's text, native or
    OCR'd, before it's classified, boundary-checked, or chunked.

    Deliberately conservative: fixes whitespace/encoding noise without
    touching content words or structural markers like "Page 2 of 2",
    since detect_document_boundary() still needs those intact.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFKC", text)

    def _is_junk_char(ch: str) -> bool:
        if ch in ("\n", "\t"):
            return False
        if ch == "�":
            return True
        # Not str.isprintable() -- it flags '\t' too, stripping tabs before
        # the whitespace-collapsing regex below can turn them into spaces.
        return unicodedata.category(ch).startswith("C")

    text = "".join(ch for ch in text if not _is_junk_char(ch))

    # Collapse horizontal whitespace runs, but keep line breaks meaningful.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # Drop speckle lines: short, and containing no letters/digits at all
    # (e.g. a lone "|" or "." picked up from a scanner artifact).
    lines = [
        line for line in lines
        if line == "" or len(line) > 3 or re.search(r"[A-Za-z0-9]", line)
    ]

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse 3+ blank lines to 1
    return cleaned.strip()


def strip_repeated_boilerplate(text: str) -> str:
    """Remove page-footer/header boilerplate from a *joined, multi-page*
    logical document, right before it's chunked and embedded.

    Only called after boundary detection has already run -- patterns
    like "Page 2 of 2" are exactly what detect_document_boundary() looks
    for, so they must survive on the per-page text. Once boundaries are
    decided, though, these lines are pure noise for retrieval.
    """
    if not text:
        return text

    patterns = [
        r"^\s*Page\s+\d+\s+of\s+\d+\s*$",   # "Page 2 of 2"
        r"^\s*Page\s+\d+\s*$",              # "Page 2"
        r"^\s*-\s*\d+\s*-\s*$",             # "- 2 -"
        r"^\s*\d+\s*$",                     # bare page number on its own line
    ]
    combined = re.compile("|".join(patterns), re.IGNORECASE)

    lines = [line for line in text.split("\n") if not combined.match(line)]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _deskew_grayscale(gray_img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect and correct rotation in a grayscale scanned-page image.

    Uses Otsu thresholding to isolate ink pixels, then fits a minimum-area
    rectangle around them to estimate the skew angle, and rotates the
    image to compensate. Returns the (possibly) rotated image and the
    detected angle in degrees.
    """
    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if coords.shape[0] < 20:
        # Not enough ink pixels to estimate an angle reliably (e.g. a
        # near-blank page) -- skip deskewing rather than guess.
        return gray_img, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Don't bother rotating for negligible skew -- avoids introducing
    # interpolation blur on pages that are already straight.
    if abs(angle) < 0.5:
        return gray_img, angle

    (h, w) = gray_img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_img, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def preprocess_scanned_page(pil_img) -> Tuple["Image.Image", float]:
    """Clean up a rendered scanned page before OCR.

    Pipeline: grayscale -> denoise -> deskew -> contrast (CLAHE) ->
    binarize (Otsu). Returns a PIL image ready for pytesseract, plus the
    detected skew angle.
    """
    from PIL import Image

    gray = np.array(pil_img.convert("L"))

    # Denoise: removes scanner/JPEG speckle without blurring text edges
    # as much as a plain Gaussian blur would.
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Deskew: rotated scans confuse Tesseract's line segmentation badly.
    gray, skew_angle = _deskew_grayscale(gray)

    # Contrast: CLAHE (adaptive histogram equalization) evens out
    # lighting across the page better than a global contrast stretch.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Binarize: Otsu picks the threshold automatically per-page, giving
    # Tesseract clean black text on white background regardless of the
    # original scan's exposure.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return Image.fromarray(binary), skew_angle


def ocr_with_confidence(pil_img) -> Tuple[str, float]:
    """Run Tesseract and return both the extracted text and its mean
    per-word confidence (0-100), instead of blindly trusting the output."""
    import pytesseract

    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    words, confidences = [], []
    for word, conf in zip(data["text"], data["conf"]):
        if word.strip():
            words.append(word)
            conf_val = int(conf) if str(conf).lstrip("-").isdigit() else -1
            if conf_val != -1:
                confidences.append(conf_val)

    text = " ".join(words)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, mean_confidence


def extract_and_analyze_pdf(pdf_file) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    """
    Extract text from PDF and perform intelligent document analysis.
    Returns both page-level info and logical document groupings.
    Supports various file types including scanned PDFs with OCR.
    """
    print("Starting PDF extraction and analysis...")

    if isinstance(pdf_file, dict) and "content" in pdf_file:
        doc = fitz.open(stream=pdf_file["content"], filetype="pdf")
    elif hasattr(pdf_file, "read"):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_file)

    pages_info = []
    for i, page in enumerate(doc):
        text = extract_page_text(page)
        was_ocr = False
        ocr_confidence = None

        if not text.strip():
            print(f"  Page {i}: No text found, attempting OCR...")
            try:
                from PIL import Image

                # Render at 2x zoom (~144 DPI instead of ~72 DPI) -- the
                # single biggest lever for OCR accuracy before any other
                # preprocessing even runs.
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                cleaned_img, skew_angle = preprocess_scanned_page(img)
                text, ocr_confidence = ocr_with_confidence(cleaned_img)
                was_ocr = True

                conf_note = f", confidence {ocr_confidence:.0f}%"
                skew_note = f", corrected {skew_angle:.1f}° skew" if abs(skew_angle) >= 0.5 else ""
                print(f"  Page {i}: OCR extracted {len(text)} characters{conf_note}{skew_note}")

                if ocr_confidence < MIN_OCR_CONFIDENCE:
                    print(f"  Page {i}: LOW OCR CONFIDENCE ({ocr_confidence:.0f}%) -- "
                          f"text may be unreliable, consider manual review")
            except Exception as e:
                print(f"  Page {i}: OCR failed - {e}")
                text = ""

        text = clean_extracted_text(text)

        pages_info.append(PageInfo(
            page_num=i, text=text, was_ocr=was_ocr, ocr_confidence=ocr_confidence
        ))

    doc.close()

    if not pages_info:
        raise ValueError("No text could be extracted from PDF")

    print(f"Extracted {len(pages_info)} pages")

    print("Analyzing document structure...")
    logical_docs = []
    current_doc_type = None
    current_doc_pages = []
    doc_counter = 0

    for i, page_info in enumerate(pages_info):
        page_text_reliable = len(page_info.text.strip()) >= MIN_TEXT_LENGTH

        if i == 0:
            if page_text_reliable:
                current_doc_type = classify_document_type(page_info.text)
            else:
                current_doc_type = "Unclassified (extraction failed)"
                print(f"  Page {i}: Text too short/unreliable to classify "
                      f"({len(page_info.text.strip())} chars) - flagged as {current_doc_type}")
            page_info.doc_type = current_doc_type
            page_info.page_in_doc = 0
            current_doc_pages = [page_info]
            print(f"  Page {i}: New document detected - {current_doc_type}")
        else:
            if not page_text_reliable or len(pages_info[i - 1].text.strip()) < MIN_TEXT_LENGTH:
                is_same = True
            else:
                prev_text = pages_info[i - 1].text
                is_same = detect_document_boundary(prev_text, page_info.text, current_doc_type)

            if is_same:
                page_info.doc_type = current_doc_type
                page_info.page_in_doc = len(current_doc_pages)
                current_doc_pages.append(page_info)
            else:
                logical_doc = LogicalDocument(
                    doc_id=f"doc_{doc_counter}",
                    doc_type=current_doc_type,
                    page_start=current_doc_pages[0].page_num,
                    page_end=current_doc_pages[-1].page_num,
                    text=strip_repeated_boilerplate("\n\n".join([p.text for p in current_doc_pages]))
                )
                logical_docs.append(logical_doc)
                doc_counter += 1

                if page_text_reliable:
                    current_doc_type = classify_document_type(page_info.text)
                else:
                    current_doc_type = "Unclassified (extraction failed)"
                    print(f"  Page {i}: Text too short/unreliable to classify "
                          f"({len(page_info.text.strip())} chars) - flagged as {current_doc_type}")
                page_info.doc_type = current_doc_type
                page_info.page_in_doc = 0
                current_doc_pages = [page_info]
                print(f"  Page {i}: New document detected - {current_doc_type}")

    if current_doc_pages:
        logical_doc = LogicalDocument(
            doc_id=f"doc_{doc_counter}",
            doc_type=current_doc_type,
            page_start=current_doc_pages[0].page_num,
            page_end=current_doc_pages[-1].page_num,
            text=strip_repeated_boilerplate("\n\n".join([p.text for p in current_doc_pages]))
        )
        logical_docs.append(logical_doc)

    print(f"Identified {len(logical_docs)} logical documents")
    for ld in logical_docs:
        print(f"   - {ld.doc_type}: Pages {ld.page_start}-{ld.page_end}")

    return pages_info, logical_docs
