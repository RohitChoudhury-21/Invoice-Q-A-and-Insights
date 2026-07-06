import sys
from pathlib import Path

# Add project root so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_processor import extract_text_from_pdf

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_pdf_text.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    text = extract_text_from_pdf(pdf_path)
    print(f"\n{'='*60}\n{pdf_path.name}\n{'='*60}\n")
    print(text if text else "(No extractable text — likely scanned image)")

if __name__ == "__main__":
    main()