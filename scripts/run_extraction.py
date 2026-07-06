import logging
import json
import sys
from pathlib import Path

# Add the project root to sys.path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_processor import extract_text_from_pdf
from app.extractor import extract_invoice_fields
from app.llm import LLMUnavailable

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    invoices_dir = Path("data/invoices_corpus")
    if not invoices_dir.exists():
        logger.error(f"Invoices folder not found: {invoices_dir}")
        sys.exit(1)

    pdf_files = sorted(invoices_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files.")

    results = []
    # Use tqdm if available, else simple loop
    iterator = tqdm(pdf_files, desc="Extracting invoices") if HAS_TQDM else pdf_files
        # Temporary: process only first 5 PDFs for testing
    pdf_files = pdf_files[:5]

    for pdf_path in iterator:
        invoice_name = pdf_path.stem
        logger.info(f"Processing {pdf_path.name}...")

        # Step 1: Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.warning(f"{pdf_path.name} has no extractable text. Skipping.")
            continue

        # Step 2: Extract fields (LLM + retry + fallback)
        try:
            invoice, path_taken = extract_invoice_fields(text)
            invoice_dict = invoice.model_dump()
            invoice_dict["source_file"] = pdf_path.name
            invoice_dict["extraction_path"] = path_taken
            results.append(invoice_dict)
        except Exception as e:
            logger.error(f"Complete extraction failure for {pdf_path.name}: {e}")
            continue

    # Save results
    output_path = Path("results/extractions.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} extractions to {output_path}")

if __name__ == "__main__":
    main()