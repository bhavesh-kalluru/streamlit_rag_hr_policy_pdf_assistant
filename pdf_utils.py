from io import BytesIO
from typing import Optional
from pypdf import PdfReader

def pdf_bytes_to_text(pdf_bytes: bytes, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF (bytes) using pypdf. 
    Note: Works for digitally selectable text; does not OCR scanned images.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    text_parts = []
    pages = reader.pages
    n = len(pages) if max_pages is None else min(max_pages, len(pages))
    for i in range(n):
        try:
            t = pages[i].extract_text() or ""
        except Exception:
            t = ""
        if t:
            text_parts.append(t.strip())
    return "\n\n".join(text_parts).strip()
