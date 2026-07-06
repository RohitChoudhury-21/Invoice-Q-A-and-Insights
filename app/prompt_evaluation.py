import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

from app.pdf_processor import extract_text_from_pdf
from app.extractor import extract_invoice_fields
from app.prompts import (
    get_extraction_prompt_v1,
    get_extraction_prompt_v2,
    get_extraction_prompt_v3,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """Load ground truth CSV with proper handling of JSON line_items."""
    df = pd.read_csv(csv_path)
    # Parse the line_items string back into Python list of dicts
    df["line_items_gt"] = df["line_items"].apply(json.loads)
    return df

def compare_invoices(gt_row: pd.Series, extracted: dict) -> dict:
    """Compare ground truth to extracted invoice and return a dict of matches."""
    # We'll compare as strings for simplicity; for date and amount we could normalize later.
    matches = {}
    for field in ["vendor", "invoice_number", "invoice_date", "total_amount", "currency"]:
        gt_val = str(gt_row[field]) if pd.notna(gt_row[field]) else ""
        ext_val = str(extracted.get(field, "")) if extracted.get(field) else ""
        matches[field] = (gt_val.strip() == ext_val.strip())
    # Compare line_items: we'll compare the sorted list of dicts as JSON strings
    gt_items = gt_row["line_items_gt"]
    ext_items = extracted.get("line_items", [])
    # Normalize both to sorted list of dicts
    gt_sorted = sorted(gt_items, key=lambda x: x["description"])
    ext_sorted = sorted(ext_items, key=lambda x: x["description"])
    matches["line_items"] = (json.dumps(gt_sorted) == json.dumps(ext_sorted))
    # Overall match: all fields must match
    matches["all_correct"] = all(matches.values())
    return matches

def evaluate_prompt(prompt_fn, df_gt: pd.DataFrame, invoices_dir: Path) -> pd.DataFrame:
    """Run extraction for all invoices in ground truth and return scores."""
    results = []
    for _, row in df_gt.iterrows():
        source_file = row["source_file"]
        pdf_path = invoices_dir / source_file
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"No text from {source_file}")
            continue
        try:
            invoice, path = extract_invoice_fields(text, prompt_fn=prompt_fn)
            ext_dict = invoice.model_dump()
            matches = compare_invoices(row, ext_dict)
            results.append({
                "source_file": source_file,
                "path": path,
                **matches
            })
        except Exception as e:
            logger.error(f"Extraction failed for {source_file}: {e}")
            # Mark all as False
            results.append({
                "source_file": source_file,
                "path": "error",
                "vendor": False, "invoice_number": False, "invoice_date": False,
                "total_amount": False, "currency": False, "line_items": False,
                "all_correct": False
            })
    return pd.DataFrame(results)

def main():
    gt_path = Path("data/ground_truth.csv")
    invoices_dir = Path("data/invoices_corpus")
    df_gt = load_ground_truth(gt_path)
    logger.info(f"Loaded ground truth for {len(df_gt)} invoices.")

    prompts = {
        "v1_plain": get_extraction_prompt_v1,
        "v2_descriptions": get_extraction_prompt_v2,
        "v3_example": get_extraction_prompt_v3,
    }

    all_scores = []
    for name, prompt_fn in prompts.items():
        logger.info(f"Evaluating prompt: {name}")
        df_results = evaluate_prompt(prompt_fn, df_gt, invoices_dir)
        # Calculate scores
        overall_acc = df_results["all_correct"].mean()
        per_field = {col: df_results[col].mean() for col in
                     ["vendor", "invoice_number", "invoice_date", "total_amount", "currency", "line_items"]}
        logger.info(f"{name}: Overall accuracy = {overall_acc:.2%}")
        row = {"prompt_version": name, "overall_accuracy": overall_acc}
        row.update(per_field)
        all_scores.append(row)

    scores_df = pd.DataFrame(all_scores)
    print("\nPrompt Evaluation Scores:")
    print(scores_df.to_string(index=False))
    scores_df.to_csv("results/prompt_scores.csv", index=False)
    logger.info("Saved scores to results/prompt_scores.csv")

if __name__ == "__main__":
    main()