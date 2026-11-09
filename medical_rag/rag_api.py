from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.rag import answer_question

app = FastAPI(title="Medical RAG API")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    print('reached message to rag')
    try:
        answer = answer_question(payload.question)
        print(answer)
        return {"answer": answer}
    except Exception:
        import logging
        logging.exception("Medical RAG request failed")
        raise HTTPException(
            status_code=500,
            detail="The medical assistant could not process this question."
        )

