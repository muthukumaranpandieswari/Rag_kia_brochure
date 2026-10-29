from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag import KiaBrochureRAG

app = FastAPI(title="Kia Brochure RAG Chat", version="1.0.0")
rag = KiaBrochureRAG()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/")
def home():
    return {"message": "Kia Brochure RAG is running"}


@app.get("/health")
def health():
    try:
        rag.connect()
        return {"status": "ok", "weaviate": "connected"}
    except Exception as exc:  # pragma: no cover
        return {"status": "degraded", "error": str(exc)}


@app.post("/api/ingest")
def ingest_brochures():
    try:
        count = rag.ingest_brochures()
        return {"status": "ok", "documents_added": count}
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "message": str(exc)}


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        matches = rag.query(request.question, top_k=request.top_k)
        answer = rag.build_answer_from_context(request.question, matches)
        return {"answer": answer, "matches": matches}
    except Exception as exc:  # pragma: no cover
        return {"answer": f"Something went wrong: {exc}", "matches": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
