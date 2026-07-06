import logging
import json
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_extractions(json_path: Path = Path("results/extractions.json")) -> pd.DataFrame:
    """Load extraction results and normalise nested fields."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data, max_level=1)
    return df

def load_exchange_rates(csv_path: Path = Path("data/exchange_rates.csv")) -> dict:
    """Return a dict mapping currency code to USD rate."""
    rates_df = pd.read_csv(csv_path)
    return dict(zip(rates_df["currency"], rates_df["rate_to_usd"]))

def parse_date(date_str: str) -> str:
    """Attempt to parse a date string into ISO format (YYYY-MM-DD)."""
    if not isinstance(date_str, str) or not date_str.strip():
        return ""
    # Try common formats
    for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # If no format matches, return original (but we'll flag it)
    return date_str.strip()

def clean_vendor(vendor: str) -> str:
    """Basic vendor name cleaning: strip, uppercase first letters."""
    if not vendor:
        return ""
    return vendor.strip().title()

def flag_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """Add flag columns for failed extractions, missing fields, etc."""
    # Flag missing critical fields
    df["flag_missing_total"] = df["total_amount"].isna() | (df["total_amount"] == 0.0)
    df["flag_missing_currency"] = df["currency"].isna() | (df["currency"].astype(str).str.strip() == "")
    df["flag_missing_vendor"] = df["vendor"].isna() | (df["vendor"].astype(str).str.strip() == "")
    df["flag_failed"] = df["flag_missing_total"] | df["flag_missing_currency"] | df["flag_missing_vendor"]
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact and content duplicates."""
    before = len(df)
    # Drop rows where all key fields are identical (exact duplicate)
    key_cols = ["vendor", "invoice_number", "invoice_date", "total_amount", "currency"]
    df = df.drop_duplicates(subset=key_cols)
    exact_removed = before - len(df)
    logger.info(f"Removed {exact_removed} exact duplicates.")

    # Content duplicate: same cleaned fields but different file (keep first)
    before = len(df)
    df = df.sort_values("source_file")
    df = df.drop_duplicates(subset=key_cols, keep="first")
    content_removed = before - len(df)
    logger.info(f"Removed {content_removed} content duplicates (same data, different file).")
    return df

def convert_to_usd(df: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """Convert total_amount and line_items amounts to USD."""
    df["currency"] = df["currency"].str.upper().str.strip()
    df["rate"] = df["currency"].map(rates).fillna(1.0)  # default to 1.0 if currency not found
    df["total_amount_usd"] = df["total_amount"] * df["rate"]
    # Convert line items (if any)
    # We'll leave line_items as they are, but we could convert each item's amount later if needed.
    return df

def check_line_item_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Flag invoices where sum of line items doesn't match total (within 1 cent)."""
    mismatches = []
    for idx, row in df.iterrows():
        items = row.get("line_items", [])
        if items:
            if isinstance(items, str):
                try:
                    items = json.loads(items)
                except json.JSONDecodeError:
                    items = []
            items_sum = sum(item.get("amount", 0.0) for item in items)
            if abs(items_sum - row["total_amount"]) > 0.01:
                mismatches.append(row.get("invoice_number", idx))
                df.at[idx, "flag_line_total_mismatch"] = True
        df.at[idx, "flag_line_total_mismatch"] = df.at[idx, "flag_line_total_mismatch"] if "flag_line_total_mismatch" in df.columns else False
    if mismatches:
        logger.info(f"Found {len(mismatches)} invoices with line‑item total mismatches: {mismatches}")
    return df

def main():
    logger.info("Loading extraction results...")
    df = load_extractions()
    logger.info(f"Raw extractions: {len(df)} rows.")

    # Clean fields
    df["vendor"] = df["vendor"].apply(clean_vendor)
    df["invoice_date"] = df["invoice_date"].apply(parse_date)
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")

    # Flag
    df = flag_invoices(df)

    # Duplicates
    df = remove_duplicates(df)

    # Currency conversion
    rates = load_exchange_rates()
    df = convert_to_usd(df, rates)

    # Line item total check
    df = check_line_item_totals(df)

    # Final columns
    out_cols = ["invoice_number", "vendor", "invoice_date", "total_amount", "currency",
                "total_amount_usd", "source_file", "flag_failed", "flag_line_total_mismatch", "line_items"]
    df_out = df[out_cols]

    # Save
    output_path = Path("data/clean_invoices.csv")
    df_out.to_csv(output_path, index=False)
    logger.info(f"Saved clean invoices to {output_path}. Final row count: {len(df_out)}.")

    # Print summary
    print(f"Before cleaning: {len(df)} rows.")
    print(f"After cleaning: {len(df_out)} rows.")
    print(f"Flagged failed extractions: {df_out['flag_failed'].sum()}")
    print(f"Line‑item mismatches: {df_out['flag_line_total_mismatch'].sum()}")
    dropped_flags = df[df['flag_failed']].copy()
    if not dropped_flags.empty:
        print("Samples of flagged/failed invoices:")
        print(dropped_flags[['invoice_number', 'source_file', 'vendor']].head(10))

if __name__ == "__main__":
    main()