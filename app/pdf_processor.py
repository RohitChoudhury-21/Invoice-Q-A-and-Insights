import pdfplumber
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file.
    Returns an empty string if the PDF contains no extractable text (e.g., scanned images).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path.name} (likely scanned image).")
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path.name}: {e}")
        return ""