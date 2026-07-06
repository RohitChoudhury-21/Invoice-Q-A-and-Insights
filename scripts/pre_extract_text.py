import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pdf_processor import extract_text_from_pdf
from tqdm import tqdm

def main():
    invoices_dir = Path("data/invoices_corpus")
    output_dir = Path("data/extracted_text")
    output_dir.mkdir(exist_ok=True)

    pdf_files = sorted(invoices_dir.glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, desc="Extracting text"):
        text = extract_text_from_pdf(pdf_path)
        out_path = output_dir / f"{pdf_path.stem}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text if text else "")

    print(f"Done. Text files saved to {output_dir}")

if __name__ == "__main__":
    main()