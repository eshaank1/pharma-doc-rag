"""
Generates synthetic test PDFs for the eval harness. Run once, or whenever
a fixture's content needs to change:

    python tests/generate_fixtures.py

Outputs land in tests/fixtures/, alongside the real pharma_blob_sample.pdf
(which is not generated -- see Task 4).
"""
import io
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _text(page, x, y, s, size=10, bold=False):
    page.insert_text((x, y), s, fontsize=size, fontname="hebo" if bold else "helv")


def generate_bordered_table():
    doc = fitz.open()
    page = doc.new_page()
    _text(page, 50, 50, "CERTIFICATE OF QUALITY", size=14, bold=True)

    rows = [
        ("Lot Number", "Manufacture Date", "Expiration Date"),
        ("LOT-1001", "2025-01-15", "2027-01-15"),
        ("LOT-1002", "2025-02-10", "2027-02-10"),
        ("LOT-1003", "2025-03-05", "2027-03-05"),
    ]
    x0, y0 = 50, 90
    col_w = [120, 120, 120]
    row_h = 20
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            rect = fitz.Rect(x, y0 + r * row_h, x + col_w[c], y0 + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((x + 3, y0 + r * row_h + 14), cell, fontsize=9)
            x += col_w[c]

    doc.save(FIXTURES_DIR / "bordered_table.pdf")


def generate_borderless_table():
    doc = fitz.open()
    page = doc.new_page()
    _text(page, 50, 50, "PACKAGING SPECIFICATION", size=14, bold=True)

    rows = [
        ("Component", "Material", "Qty"),
        ("Blister Tray", "PETG", "500"),
        ("Lid Film", "Tyvek", "500"),
        ("Carton", "Corrugate", "50"),
    ]
    y = 90
    for row in rows:
        x = 50
        for cell in row:
            _text(page, x, y, cell, size=9)
            x += 120
        y += 20

    doc.save(FIXTURES_DIR / "borderless_table.pdf")


def generate_cover_letter():
    doc = fitz.open()
    page = doc.new_page()
    letter = (
        "To Whom It May Concern,\n\n"
        "This letter certifies that the enclosed materials meet all applicable\n"
        "specifications for pharmaceutical packaging components as described in\n"
        "the attached Certificate of Quality and Packaging Specification.\n\n"
        "Please contact our Quality Assurance department with any questions\n"
        "regarding storage conditions or handling requirements for this shipment.\n\n"
        "Sincerely,\n"
        "Jane Doe\n"
        "Quality Assurance Manager\n"
    )
    page.insert_text((50, 80), letter, fontsize=11)
    doc.save(FIXTURES_DIR / "cover_letter.pdf")


def _rasterize_page_to_image_pdf(source_doc, out_path, noise_amp=0):
    """Render a text-based page to a grayscale image, optionally add noise,
    and save it as a new PDF with NO text layer -- so page.get_text()
    returns empty and the real OCR path in extract_and_analyze_pdf() runs,
    simulating a scanned document."""
    src_page = source_doc[0]
    pix = src_page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")

    if noise_amp:
        arr = np.array(img).astype(np.int16)
        rng = np.random.default_rng(42)  # fixed seed: reproducible OCR behavior
        arr += rng.integers(-noise_amp, noise_amp, size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out_page.insert_image(out_page.rect, stream=buf.getvalue())
    out_doc.save(out_path)


def generate_scanned_certificate():
    doc = fitz.open()
    page = doc.new_page()
    _text(page, 50, 50, "CERTIFICATE OF QUALITY", size=14, bold=True)
    rows = [
        ("Lot Number", "Manufacture Date", "Expiration Date"),
        ("LOT-2001", "2025-04-01", "2027-04-01"),
    ]
    x0, y0 = 50, 90
    col_w = [120, 120, 120]
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            page.insert_text((x + 3, y0 + r * 20 + 14), cell, fontsize=9)
            x += col_w[c]

    _rasterize_page_to_image_pdf(doc, FIXTURES_DIR / "scanned_certificate.pdf", noise_amp=0)


def generate_scanned_bse_tse():
    doc = fitz.open()
    page = doc.new_page()
    _text(page, 50, 50, "BSE/TSE DECLARATION", size=14, bold=True)
    _text(page, 50, 90, "Manufacturer: Cytiva Sweden AB", size=10)
    _text(page, 50, 110, "This product does not contain any materials of animal origin.", size=10)
    _text(page, 50, 130, "Assessment Reference: BSE-RA-2025-TEST-001", size=10)

    # noise_amp=45 was hand-validated to produce a realistic, reproducible
    # OCR degradation (~93% confidence, "contain" misread as "caniain")
    # without breaking OCR entirely -- this pipeline's denoise+CLAHE+Otsu
    # preprocessing is a hard cliff (works cleanly up to ~45-46, falls to
    # 0% confidence at ~48+), so don't raise this value without re-checking
    # actual OCR output, not just assuming "more noise = more degraded."
    _rasterize_page_to_image_pdf(doc, FIXTURES_DIR / "scanned_bse_tse.pdf", noise_amp=45)


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_bordered_table()
    generate_borderless_table()
    generate_cover_letter()
    generate_scanned_certificate()
    generate_scanned_bse_tse()
    print(f"Generated 5 fixtures in {FIXTURES_DIR}")
