import logging
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from app.embeddings import embed_texts
from app.vector_store import upsert_chunks, search as vector_search

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP = 50 # characters overlap between chunks

def chunk_text(text: str, source_id: str) -> List[Dict[str, object]]:
    """
    Split text into overlapping chunks and return a list of dicts with:
    - id: unique chunk ID (invoiceNumber#index)
    - text: chunk content
    - metadata: dict with invoice_number
    """
    if not text.strip():
        return []

    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "id": f"{source_id}#{idx}",
                "text": chunk,
                "metadata": {"invoice_number": source_id}
            })
            idx += 1
        # slide forward by chunk_size - overlap
        start += (CHUNK_SIZE - CHUNK_OVERLAP)
        if start >= len(text):
            break
    return chunks

def load_invoice_number_map(gt_csv: Path = Path("data/ground_truth.csv")) -> dict:
    """Return a dict mapping PDF stem (e.g., '1') to invoice number (e.g., 'INV-24990')."""
    if not gt_csv.exists():
        logger.warning("ground_truth.csv not found. Using file stems as invoice numbers.")
        return {}
    df = pd.read_csv(gt_csv)
    # source_file column contains something like '1.pdf', extract stem
    df["stem"] = df["source_file"].str.replace(".pdf", "", regex=False)
    return dict(zip(df["stem"], df["invoice_number"]))

def build_index(text_dir: Path = Path("data/extracted_text")):
    """
    Read all .txt files, chunk, embed, and upsert into ChromaDB.
    """
    txt_files = sorted(text_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files to index.")

    inv_map = load_invoice_number_map()   # load once before the loop

    all_chunk_ids = []
    all_chunk_texts = []
    all_embeddings = []
    all_metadatas = []

    for txt_file in txt_files:
        invoice_number = inv_map.get(txt_file.stem, txt_file.stem)  # e.g., '1' -> 'INV-24990'
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, invoice_number)
        for ch in chunks:
            all_chunk_ids.append(ch["id"])
            all_chunk_texts.append(ch["text"])
            all_metadatas.append(ch["metadata"])

    if not all_chunk_ids:
        logger.warning("No chunks to index.")
        return

    logger.info(f"Embedding {len(all_chunk_ids)} chunks...")
    all_embeddings = embed_texts(all_chunk_texts)

    # Upsert into vector store
    upsert_chunks(
        chunk_ids=all_chunk_ids,
        chunk_texts=all_chunk_texts,
        embeddings=all_embeddings,
        metadatas=all_metadatas
    )
    logger.info("Indexing complete.")

def search(question: str, n_results: int = 4, filter: Optional[Dict] = None) -> List[Dict]:
    """
    Embed the question, search ChromaDB, optionally filter by metadata.
    """
    q_embedding = embed_texts([question])[0]
    hits = vector_search(q_embedding, n_results=n_results, filter=filter)
    return hits