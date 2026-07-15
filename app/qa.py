# app/qa.py
import logging
import json
import re
from typing import Callable, List, Dict, Any, Optional

from app.retriever import search
from app.llm import generate
from app.extractor import _extract_json

logger = logging.getLogger(__name__)

def extract_invoice_number(question: str) -> Optional[str]:
    """Find an invoice number pattern (e.g., INV-24990) in the question."""
    match = re.search(r'INV[-\s]*\d{1,7}', question, re.IGNORECASE)
    if match:
        return match.group(0).replace(" ", "-")  # normalise
    # Also try "invoice 24990" style
    match = re.search(r'invoice[-\s]*(\d{1,7})', question, re.IGNORECASE)
    if match:
        return f"INV-{match.group(1)}"
    return None

def build_qa_prompt(question: str, chunks: List[Dict]) -> List[Dict[str, str]]:
    chunk_texts = "\n\n".join(
        f"[Invoice {ch['metadata']['invoice_number']}] {ch['text']}" for ch in chunks
    )
    system = (
        "You are a helpful assistant that answers questions based ONLY on the provided invoice snippets. "
        "If the answer is not found in the snippets, respond with exactly 'I don't know'. "
        "If the question asks for a person (e.g., CEO, founder, employee) or a role and it is NOT explicitly stated in the snippets, "
        "you must answer 'I don't know'. "
        "You MUST return a JSON object with exactly two keys: "
        "'answer' (string) and 'sources' (list of invoice numbers used). "
        "Only include invoice numbers from the snippets you actually used. "
        "Return nothing else."
    )
    user = f"Question: {question}\n\nSnippets:\n{chunk_texts}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

def is_faithful(answer: str, chunks: List[Dict]) -> bool:
    """Check that the answer is grounded in the retrieved chunks."""
    if answer.strip().lower() == "i don't know":
        return True

    all_chunk_text = " ".join(ch["text"] for ch in chunks)
    all_chunk_text_clean = all_chunk_text.replace(",", "")

    # 1. Numeric check: any number in the answer must appear in the chunks
    answer_clean = answer.replace(",", "")
    numbers = re.findall(r'\d+\.?\d*', answer_clean)
    for num in numbers:
        if num not in all_chunk_text_clean:
            logger.warning(f"Number '{num}' not found in chunks.")
            return False

    # 2. Named‑entity check: any capitalised word (potential name, place, etc.)
    #    longer than 2 chars and not a common stop‑word must appear in the chunks.
    potential_entities = re.findall(r'\b[A-Z][a-z]{2,}\b', answer)
    stop_words = {"The", "What", "When", "Where", "Who", "How", "This", "That", "There",
                  "They", "Their", "Will", "Would", "Could", "Should", "Invoice", "Total",
                  "Amount", "Date", "Currency", "Vendor", "Number", "Issue", "Due", "Grand"}
    for word in potential_entities:
        if word not in stop_words and word.lower() not in all_chunk_text.lower():
            logger.warning(f"Entity '{word}' not found in chunks.")
            return False

    return True

def ask(question: str, n_chunks: int = 4, prompt_fn: Callable = None) -> Dict[str, Any]:
    if prompt_fn is None:
        prompt_fn = build_qa_prompt_v2  

    # 1. Check for invoice filter
    invoice_filter = None
    inv_num = extract_invoice_number(question)
    if inv_num:
        invoice_filter = {"invoice_number": inv_num}
        logger.info(f"Applying filter for invoice: {inv_num}")

    # 2. Retrieve chunks FIRST
    chunks = search(question, n_results=n_chunks, filter=invoice_filter)
    if not chunks:
        return {"answer": "I don't know", "sources": [], "context_found": False}

    # 3. Build prompt using the retrieved chunks
    messages = prompt_fn(question, chunks)   # <-- chunks now exists
    raw_output = generate(messages, max_tokens=256)

    try:
        json_str = _extract_json(raw_output)
        data = json.loads(json_str)
        answer = data.get("answer", "I don't know")
        sources = data.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        sources = [str(s) for s in sources if s]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse LLM answer: {e}")
        answer = "I don't know"
        sources = []

    # 3.5 Post‑check for person questions (who, what is the CEO, etc.)
    person_keywords = r'\b(?:who|ceo|founder|employee|auditor|president|director|owner|manager|staff)\b'
    if re.search(person_keywords, question, re.IGNORECASE):
        # Look for a likely person name: two consecutive capitalised words not in the stop‑list
        person_pattern = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', answer)
        invoice_stop_words = {"Massive Dynamic", "Acme Corp", "Initech Inc", "Globex LLC",
                              "Vandelay Industries", "Cyberdyne Systems", "Wonka Industries",
                              "Nakatomi Trading", "Gekko And Co", "Aperture Labs", "Pied Piper Llc"}
        possible_names = [name for name in person_pattern if name.lower() not in
                          [s.lower() for s in invoice_stop_words]]
        if not possible_names:
            logger.info("Question asks for a person but no person name found – forcing I don't know.")
            answer = "I don't know"
            sources = []
            # (context_found will be set later)

    # 4. Faithfulness check
    if not is_faithful(answer, chunks):
        answer = "I don't know"
        sources = []
        context_found = False
    else:
        context_found = (answer.strip().lower() != "i don't know")

    return {
        "answer": answer,
        "sources": sources,
        "context_found": context_found
    }

def build_qa_prompt_v2(question: str, chunks: List[Dict]) -> List[Dict[str, str]]:
    chunk_texts = "\n\n".join(
        f"[Invoice {ch['metadata']['invoice_number']}] {ch['text']}" for ch in chunks
    )
    system = (
        "You are an assistant that strictly answers ONLY from the provided invoice snippets. "
        "If the information is not explicitly in the snippets, say 'I don't know'. "
        "Do NOT guess, infer, or use general knowledge. "
        "If the question asks for a person, organization leader, employee count, tax rate, or any fact "
        "not present in the snippets, you MUST reply 'I don't know'. "
        "Return a JSON object with exactly two keys: 'answer' (string) and 'sources' (list of invoice numbers). "
        "Only include invoice numbers whose snippets you actually used. "
        "Return nothing else."
    )
    user = f"Question: {question}\n\nSnippets:\n{chunk_texts}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]