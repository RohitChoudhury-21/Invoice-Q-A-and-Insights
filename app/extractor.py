import logging
import json
import re
from typing import Optional, Callable, List, Dict

from pydantic import ValidationError

from app.llm import generate, LLMUnavailable
from app.schemas import Invoice
from app.prompts import get_extraction_prompt_v2 

logger = logging.getLogger(__name__)


def extract_invoice_fields(
    invoice_text: str,
    prompt_fn: Optional[Callable[[str], List[Dict[str, str]]]] = None
) -> tuple[Invoice, str]:
    """
    Extract invoice fields using the LLM, with one retry and a regex fallback.
    Returns (Invoice, path) where path is 'llm', 'retry', or 'fallback'.
    """
    # Use default prompt if none supplied
    if prompt_fn is None:
        prompt_fn = get_extraction_prompt_v2

    last_error_msg = None

    # --- First attempt ---
    try:
        invoice = _extract_with_llm(invoice_text, error_feedback=None, prompt_fn=prompt_fn)
        logger.info("Extraction succeeded on first attempt (llm).")
        return invoice, "llm"
    except (LLMUnavailable, ValidationError, json.JSONDecodeError, ValueError) as e:
        last_error_msg = str(e)
        logger.warning(f"First attempt failed: {last_error_msg}")
    except Exception as e:
        last_error_msg = str(e)
        logger.warning(f"First attempt failed with unexpected error: {last_error_msg}")

    # --- Retry with error feedback ---
    try:
        invoice = _extract_with_llm(invoice_text, error_feedback=last_error_msg, prompt_fn=prompt_fn)
        logger.info("Extraction succeeded on retry.")
        return invoice, "retry"
    except (LLMUnavailable, ValidationError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Retry failed: {e}")
    except Exception as e:
        logger.warning(f"Retry failed with unexpected error: {e}")

    # --- Regex fallback ---
    logger.info("Falling back to regex extractor.")
    invoice = regex_extract_invoice(invoice_text)
    logger.info("Extraction succeeded via regex fallback.")
    return invoice, "fallback"


def _extract_with_llm(
    invoice_text: str,
    error_feedback: Optional[str] = None,
    prompt_fn: Callable[[str], List[Dict[str, str]]] = get_extraction_prompt_v2
) -> Invoice:
    """
    Calls the LLM with the extraction prompt, optionally including error feedback.
    """
    messages = prompt_fn(invoice_text)
    if error_feedback:
        messages.append({
            "role": "user",
            "content": f"Previous attempt failed with error: {error_feedback}. "
                       f"Please correct and return only valid JSON."
        })

    raw_output = generate(messages, max_tokens=512)
    json_str = _extract_json(raw_output)
    data = json.loads(json_str)
    return Invoice.model_validate(data)


def _extract_json(text: str) -> str:
    """Extract JSON object from model output, with basic repair."""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        json_candidate = text[start:end+1]
        try:
            json.loads(json_candidate)
            return json_candidate
        except json.JSONDecodeError:
            repaired = json_candidate.replace("'", '"')
            return repaired
    return text


def regex_extract_invoice(text: str) -> Invoice:
    """Fallback regex extractor with broader pattern coverage."""
    logger.info("Using regex fallback to extract invoice fields.")

    # Vendor: first non‑empty line (often the company name)
    vendor_match = re.search(r'^(.+)$', text, re.MULTILINE)
    vendor = vendor_match.group(1).strip() if vendor_match else ""

    # Invoice number: common labels
    invoice_num = (
        _regex_field(r'(?:Billing\s*ID|Invoice\s*(?:#|No|Number))\s*[:\-\s]+\s*([A-Za-z0-9\-]+)', text)
    )

    # Date: common phrases
    date = (
        _regex_field(r'Date\s*(?:of issue|issued)?\s*[:\-\s]+\s*([\d]{1,2}[\s\w]*[\d]{4})', text)
    )

    # Total: various wordings
    total = (
        _regex_field(r'(?:Grand\s*Total|Total\s*Due|Please\s*remit|Amount\s*Due)[\s:\w]*([\d,]+\.?\d{0,2})', text)
    )

    # Currency: three‑letter code near total
    currency = (
        _regex_field(r'(?:Grand\s*Total|Total\s*Due|Please\s*remit)[\s:\w]*([A-Z]{3})', text) or
        _regex_field(r'([A-Z]{3})\s*[\d,]+\.?\d{0,2}\s*$', text)  # fallback
    )

    # Line items – harder; we’ll leave empty for now

    return Invoice(
        vendor=vendor.strip() if vendor else "",
        invoice_number=invoice_num.strip() if invoice_num else "",
        invoice_date=date.strip() if date else "",
        total_amount=float(total.replace(',', '')) if total else 0.0,
        currency=currency.strip() if currency else "",
        line_items=[]
    )


def _regex_field(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


if __name__ == "__main__":
    sample_text = """
MASSIVE DYNAMIC
Toronto | +1-557-592-8045 | billing@massive.example
For the following:
  - 12 x a data subscription at GBP 476.40 = GBP 5,716.80
  - 11 x consulting hours at GBP 581.51 = GBP 6,396.61
Bill to J. Smith, accounts payable. Tax registration VAT209289120.
Subtotal GBP 12,113.41.
Tax 8% adds GBP 969.07.
Grand Total: GBP 13,082.48.
Date of issue 01 June 2026, terms Net 15, due 31.07.2026.
Pay by transfer to IBAN DE28 5918 7276. Thank you for your business.
Billing ID: INV-24990. Customer account AC9629231. PO PO-65324.
"""
    result, path = extract_invoice_fields(sample_text)
    print(f"Path: {path}")
    print(result.model_dump_json(indent=2))