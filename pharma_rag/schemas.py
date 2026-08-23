from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PageInfo:
    """Stores information about a single page"""
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0
    was_ocr: bool = False                       # True if this page went through OCR
    ocr_confidence: Optional[float] = None       # Tesseract mean word confidence (0-100), OCR pages only


@dataclass
class LogicalDocument:
    """Represents a logical document within a PDF"""
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    chunks: List[Dict] = None


@dataclass
class ChunkMetadata:
    """Rich metadata for each chunk"""
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    embedding: Optional[np.ndarray] = None
