import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model = None

def _load_model():
    global _model
    if _model is None:
        logger.info("Loading embedding model all-MiniLM-L6-v2...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a list of 384‑dimensional embeddings for the given texts."""
    model = _load_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()