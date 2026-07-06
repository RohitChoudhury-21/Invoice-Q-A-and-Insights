import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Module‑level client – created once
_client = None
_collection = None

COLLECTION_NAME = "invoice_chunks"

def _get_client():
    global _client
    if _client is None:
        logger.info("Initializing ChromaDB client...")
        _client = chromadb.PersistentClient(
            path="vector_db/chroma",
            settings=Settings(anonymized_telemetry=False)
        )
    return _client

def get_collection():
    """Get or create the ChromaDB collection."""
    global _collection
    client = _get_client()
    if _collection is None:
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{COLLECTION_NAME}' ready. Count: {_collection.count()}")
    return _collection

def upsert_chunks(
    chunk_ids: List[str],
    chunk_texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, str]]
):
    """
    Insert or update chunks. Uses the 'id' field to avoid duplicates.
    """
    collection = get_collection()
    collection.upsert(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    logger.info(f"Upserted {len(chunk_ids)} chunks. New total: {collection.count()}")

def search(query_embedding: List[float], n_results: int = 4, filter: Optional[Dict] = None) -> List[Dict]:
    """
    Search the collection with optional metadata filtering.
    filter: e.g. {"invoice_number": "INV-24990"}
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filter,  # <-- pass filter here
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    if results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = (2.0 - distance) / 2.0
            hits.append({
                "id": chunk_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": round(similarity, 4)
            })
    return hits