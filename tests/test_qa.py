import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.qa import ask

def test_answerable_question():
    """Question about a specific invoice that exists in the corpus."""
    result = ask("What is the total amount on invoice INV-24990?")
    assert result["context_found"] is True
    assert "13082.48" in result["answer"].replace(",", "")    
    assert len(result["sources"]) > 0

def test_unanswerable_question():
    """Question that cannot be answered from any invoice."""
    result = ask("What is the CEO of Massive Dynamic?")
    print("DEBUG answer:", repr(result["answer"]))
    print("DEBUG sources:", result["sources"])
    print("DEBUG context_found:", result["context_found"])
    assert result["context_found"] is False
    assert result["answer"].strip().lower() == "i don't know"
    assert result["sources"] == []