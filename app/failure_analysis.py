import logging
import json
from typing import Dict, Any, List, Tuple
from app.extractor import extract_invoice_fields, regex_extract_invoice
from app.schemas import Invoice
import pandas as pd
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def compare_outputs(llm_inv: Invoice, regex_inv: Invoice) -> Dict[str, Any]:
    """Return a dict with agreement flags and confidence."""
    # Simple string compare for vendor (case‑insensitive)
    vendor_agree = (llm_inv.vendor.strip().lower() == regex_inv.vendor.strip().lower())
    total_agree = abs(llm_inv.total_amount - regex_inv.total_amount) < 0.01
    currency_agree = (llm_inv.currency.strip().upper() == regex_inv.currency.strip().upper())
    inv_num_agree = (llm_inv.invoice_number.strip() == regex_inv.invoice_number.strip())

    all_agree = vendor_agree and total_agree and currency_agree and inv_num_agree
    confidence = "high" if all_agree else "low"
    return {
        "vendor_agree": vendor_agree,
        "total_agree": total_agree,
        "currency_agree": currency_agree,
        "inv_num_agree": inv_num_agree,
        "all_agree": all_agree,
        "confidence": confidence
    }

def analyze_case(case_name: str, invoice_text: str, expected_note: str) -> Dict:
    """Run LLM and regex extraction on a single case, return analysis."""
    # LLM extraction
    try:
        llm_inv, path = extract_invoice_fields(invoice_text)
        llm_dict = llm_inv.model_dump()
        llm_success = True
    except Exception as e:
        llm_dict = {"error": str(e)}
        llm_success = False
        llm_inv = None

    # Regex extraction
    regex_inv = regex_extract_invoice(invoice_text)
    regex_dict = regex_inv.model_dump()

    # Compare if LLM succeeded
    if llm_success:
        comparison = compare_outputs(llm_inv, regex_inv)
    else:
        comparison = {"confidence": "low", "error": "LLM failed"}

    return {
        "case": case_name,
        "llm_output": llm_dict,
        "regex_output": regex_dict,
        "comparison": comparison,
        "expected_note": expected_note
    }

def main():
    # Define hard cases using real invoice texts (extract text from PDFs or use the ones you already saw).
    # For scanned, use empty text (since pdfplumber gave empty).
    # For missing total, craft a text without a total line.
    # For foreign currency, pick an invoice with INR or EUR.
    # For near‑duplicate, take an invoice and change a few digits.

    hard_cases = [
        {
            "name": "scanned_image",
            "text": "",   # simulate no extractable text
            "expected": "Should produce empty output; regex returns mostly empty."
        },
        {
            "name": "missing_total",
            "text": (
                "VENDOR: ACME Corp\n"
                "Invoice #: INV-001\n"
                "Date: 2025-01-15\n"
                "Items:\n"
                " - widget: 10.00\n"
                " - gadget: 20.00\n"
                # deliberately no total
            ),
            "expected": "LLM may invent a total; regex returns 0.0."
        },
        {
            "name": "foreign_currency",
            "text": (
                "MASSIVE DYNAMIC\n"
                "Billing ID: INV-24990\n"
                "Date of issue 01 June 2026\n"
                "Grand Total: GBP 13,082.48\n"
                # using GBP as foreign
            ),
            "expected": "Should extract correctly."
        },
        {
            "name": "near_duplicate",
            "text": (
                "MASSIVE DYNAMIC\n"
                "Billing ID: INV-24990-DUP\n"   # slightly different number
                "Date of issue 01 June 2026\n"
                "Grand Total: GBP 13,082.48\n"
                # everything else same as original
            ),
            "expected": "LLM might confuse with the original; should capture the changed invoice number."
        }
    ]

    results = []
    for case in hard_cases:
        logger.info(f"Analyzing case: {case['name']}")
        analysis = analyze_case(case["name"], case["text"], case["expected"])
        results.append(analysis)

    # Print failure table
    print("\n=== FAILURE ANALYSIS TABLE ===")
    for r in results:
        print(f"\nCase: {r['case']}")
        print(f"  LLM output: {json.dumps(r['llm_output'], indent=4)}")
        print(f"  Regex output: {json.dumps(r['regex_output'], indent=4)}")
        print(f"  Comparison: {json.dumps(r['comparison'], indent=4)}")
        print(f"  Expected: {r['expected_note']}")

    # Trust-or-fallback rule
    print("\n=== TRUST-OR-FALLBACK RULE ===")
    print("If LLM and regex disagree on total_amount (diff > 0.01) or vendor (case-insensitive), "
          "or if the invoice text is empty (scanned), use the regex fallback.")
    print("If regex also returns mostly empty, flag for manual review or OCR.")

    print("\n=== WHEN LLM IS THE WRONG TOOL ===")
    print("Scanned image invoices: LLM cannot process images directly. Instead, "
          "use an OCR service (Tesseract, cloud OCR) to extract text first, then run extraction.")

    # --- Save results as CSV ---
    csv_rows = []
    for r in results:
        llm = r['llm_output']
        regex = r['regex_output']
        comp = r['comparison']
        row = {
            "case": r["case"],
            "llm_vendor": llm.get("vendor", ""),
            "llm_invoice_number": llm.get("invoice_number", ""),
            "llm_invoice_date": llm.get("invoice_date", ""),
            "llm_total_amount": llm.get("total_amount", ""),
            "llm_currency": llm.get("currency", ""),
            "llm_line_items": json.dumps(llm.get("line_items", [])),
            "regex_vendor": regex.get("vendor", ""),
            "regex_invoice_number": regex.get("invoice_number", ""),
            "regex_invoice_date": regex.get("invoice_date", ""),
            "regex_total_amount": regex.get("total_amount", ""),
            "regex_currency": regex.get("currency", ""),
            "confidence": comp.get("confidence", ""),
            "vendor_agree": comp.get("vendor_agree", False),
            "total_agree": comp.get("total_agree", False),
            "currency_agree": comp.get("currency_agree", False),
            "inv_num_agree": comp.get("inv_num_agree", False),
            "all_agree": comp.get("all_agree", False),
            "expected_note": r["expected_note"]
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv("results/failures.csv", index=False)
    logger.info("Failures table saved to results/failures.csv")

if __name__ == "__main__":
    main()