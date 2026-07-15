# app/api.py
from fastapi import FastAPI, Query
from app.qa import ask

app = FastAPI(title="Invoice Q&A", version="1.0.0")

@app.get("/ask")
def ask_question(question: str = Query(..., description="Your invoice question")):
    """Answer a question based on the invoice collection."""
    return ask(question)