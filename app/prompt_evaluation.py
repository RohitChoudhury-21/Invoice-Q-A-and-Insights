import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from app.pdf_processor import extract_text_from_pdf
from app.extractor import extract_invoice_fields
from app.prompts import (
    get_extraction_prompt_v1,
    get_extraction_prompt_v2,
    get_extraction_prompt_v3,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_date_iso(date_str: str) -> str:
    """Try to parse a date string into ISO format YYYY-MM-DD."""
    if not isinstance(date_str, str) or not date_str.strip():
        return ""
    for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # If no format matches, return the original stripped string
    return date_str.strip()

def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """Load ground truth CSV with proper handling of JSON line_items."""
    df = pd.read_csv(csv_path)
    # Parse the line_items string back into Python list of dicts
    df["line_items_gt"] = df["line_items"].apply(json.loads)
    return df

def compare_invoices(gt_row: pd.Series, extracted: dict) -> dict:
    """Compare ground truth to extracted invoice, normalising dates and loosely matching line items."""
    matches = {}
    
    # Normalise dates
    gt_date = parse_date_iso(str(gt_row["invoice_date"]))
    ext_date = parse_date_iso(str(extracted.get("invoice_date", "")))
    matches["invoice_date"] = (gt_date == ext_date)
    
    # Compare other text fields as before (case‑insensitive, stripped)
    for field in ["vendor", "invoice_number", "total_amount", "currency"]:
        gt_val = str(gt_row[field]).strip().lower() if pd.notna(gt_row[field]) else ""
        ext_val = str(extracted.get(field, "")).strip().lower() if extracted.get(field) else ""
        # For total_amount, compare as floats with tolerance
        if field == "total_amount":
            try:
                gt_num = float(gt_row[field])
                ext_num = float(extracted.get(field, 0))
                matches[field] = abs(gt_num - ext_num) < 0.01
            except (ValueError, TypeError):
                matches[field] = False
        else:
            matches[field] = (gt_val == ext_val)
    
    # Loose line‑item matching
    gt_items = gt_row["line_items_gt"]   # already parsed as list of dicts
    ext_items = extracted.get("line_items", [])
    if not gt_items:
        # If ground truth has no items, consider it a match if extraction also has none
        matches["line_items"] = (len(ext_items) == 0)
    else:
        # Simple matching: for each GT item, check if any extracted item has a similar description
        # and amount close enough. Give credit if the majority match.
        matched = 0
        for gt in gt_items:
            gt_desc = gt["description"].strip().lower()
            gt_amt = gt["amount"]
            for ext in ext_items:
                ext_desc = ext.get("description", "").strip().lower()
                ext_amt = ext.get("amount", 0)
                # Use a simple keyword overlap (both contain the same core word)
                if gt_desc in ext_desc or ext_desc in gt_desc:
                    if abs(gt_amt - ext_amt) < 0.01:
                        matched += 1
                        break
        # Consider a pass if at least half of the GT items matched
        matches["line_items"] = (matched >= len(gt_items) / 2)
    
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