# tests/test_search.py
from app.retriever import search

def test_search_returns_4_results():
    """Search should return 4 scored chunks with invoice metadata."""
    results = search("total amount INV-24990")
    assert len(results) == 4
    for r in results:
        assert "score" in r
        assert "metadata" in r
        assert "invoice_number" in r["metadata"]
        assert 0.0 <= r["score"] <= 1.0