import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.rag import KiaBrochureRAG

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


rag = KiaBrochureRAG()

app = FastAPI(
    title="Kia Brochure RAG",
    description="Ask questions about Kia brochure information using semantic search.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    if not settings.WEAVIATE_URL or not settings.WEAVIATE_API_KEY:
        logger.warning(
            "WEAVIATE_URL and WEAVIATE_API_KEY are not configured. The app will start "
            "without a vector store until the environment variables are set."
        )
        return

    try:
        rag.ensure_collection()
        rag.ingest_brochures()
    except Exception as exc:  # pragma: no cover - defensive logging for deployment
        logger.warning("Could not initialize brochure data store: %s", exc)


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Kia Brochure RAG</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2rem; background: #f5f5f5; }
                .card { max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
                textarea, input, button { width: 100%; padding: 0.75rem; margin-top: 0.75rem; border-radius: 8px; border: 1px solid #d0d5dd; }
                button { background: #0f172a; color: white; border: none; cursor: pointer; }
                #answer { margin-top: 1.5rem; white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Kia Brochure RAG</h1>
                <p>Ask a question about Kia vehicles, features, or brochure content.</p>
                <textarea id="question" rows="4" placeholder="Example: What are the key features of the Kia EV6?"></textarea>
                <input id="top_k" type="number" min="1" max="10" value="4" />
                <button onclick="ask()">Ask</button>
                <div id="answer"></div>
            </div>
            <script>
                async function ask() {
                    const question = document.getElementById('question').value.trim();
                    const topK = Number(document.getElementById('top_k').value || 4);
                    const answerEl = document.getElementById('answer');
                    if (!question) {
                        answerEl.textContent = 'Please enter a question.';
                        return;
                    }
                    answerEl.textContent = 'Searching...';
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question, top_k: topK })
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        answerEl.textContent = data.detail || 'Error while answering.';
                        return;
                    }
                    answerEl.textContent = data.answer || 'No answer returned.';
                }
            </script>
        </body>
        </html>
        """
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "weaviate_configured": bool(settings.WEAVIATE_URL and settings.WEAVIATE_API_KEY),
    }


@app.post("/ask")
async def ask_question(payload: AskRequest) -> Dict[str, Any]:
    if not settings.WEAVIATE_URL or not settings.WEAVIATE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="The RAG service is not configured. Set WEAVIATE_URL and WEAVIATE_API_KEY first.",
        )

    try:
        matches = rag.query(payload.question, top_k=payload.top_k)
        answer = rag.build_answer_from_context(payload.question, matches)
        return {"answer": answer, "results": matches}
    except Exception as exc:
        logger.exception("Error while querying brochure knowledge base")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc
