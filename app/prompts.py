# app/prompts.py

def get_extraction_prompt_v1(invoice_text: str) -> list[dict[str, str]]:
    """
    Builds a prompt that instructs the model to extract invoice fields as JSON.
    Returns a list of messages suitable for the LLM's chat template.
    """
    system_instruction = (
        "You are a precise invoice data extraction assistant. "
        "Extract the following fields from the invoice text and return ONLY a valid JSON object "
        "that matches this structure, with no additional text or explanation: "
        '{"vendor": "<string>", "invoice_number": "<string>", "invoice_date": "<string>", '
        '"total_amount": <float>, "currency": "<string>", '
        '"line_items": [{"description": "<string>", "amount": <float>}]}. '
        "If a field is not present, use null or an empty string as appropriate. "
        "For line items, include each distinct product or service line, using the per-unit amount "
        "or the line total as shown; do not multiply by quantity unless the invoice does so. "
        "Return nothing else."
    )

    user_message = f"Invoice text:\n{invoice_text}"

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]

def get_extraction_prompt_v2(invoice_text: str) -> list[dict[str, str]]:
    """
    Prompt with a short description of each field.
    """
    system_instruction = (
        "You are a precise invoice data extraction assistant. "
        "Extract the following fields from the invoice text and return ONLY a valid JSON object "
        "matching this structure, with no additional text:\n"
        '{"vendor": "<string>", "invoice_number": "<string>", "invoice_date": "<string>", '
        '"total_amount": <float>, "currency": "<string>", '
        '"line_items": [{"description": "<string>", "amount": <float>}]}\n'
        "Field descriptions:\n"
        "- vendor: The company or individual issuing the invoice.\n"
        "- invoice_number: The unique identifier for this invoice (e.g., INV-12345, Billing ID).\n"
        "- invoice_date: The date the invoice was issued (as it appears).\n"
        "- total_amount: The final total amount due (as a number, without currency symbol).\n"
        "- currency: The three-letter currency code (e.g., USD, EUR, GBP) shown on the invoice.\n"
        "- line_items: A list of distinct products or services, each with a description and its "
        "line total amount (the final amount for that line, including any quantity).\n"
        "If a field is missing, use null or empty string. Return nothing else."
    )
    user_message = f"Invoice text:\n{invoice_text}"
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]

def get_extraction_prompt_v3(invoice_text: str) -> list[dict[str, str]]:
    """
    Prompt that includes a full worked example of an invoice and its correct JSON.
    """
    system_instruction = (
        "You are a precise invoice data extraction assistant. "
        "Extract the fields from the invoice text and return ONLY a valid JSON object "
        "matching this structure, with no additional text:\n"
        '{"vendor": "<string>", "invoice_number": "<string>", "invoice_date": "<string>", '
        '"total_amount": <float>, "currency": "<string>", '
        '"line_items": [{"description": "<string>", "amount": <float>}]}\n\n'
        "Example:\n"
        "Invoice text:\n"
        "\"ACME Corp\n"
        "Invoice #INV-001\n"
        "Date: 15 March 2025\n"
        "Items:\n"
        "  - 3 x Widgets at $10.00 each = $30.00\n"
        "  - 2 x Gadgets at $20.00 each = $40.00\n"
        "Total: $70.00 USD\"\n\n"
        "Correct JSON output:\n"
        '{"vendor": "ACME Corp", "invoice_number": "INV-001", "invoice_date": "15 March 2025", '
        '"total_amount": 70.0, "currency": "USD", '
        '"line_items": [{"description": "Widgets", "amount": 30.0}, {"description": "Gadgets", "amount": 40.0}]}\n\n'
        "Now extract from the provided invoice text. Return nothing else."
    )
    user_message = f"Invoice text:\n{invoice_text}"
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]