import os
from typing import Any, Dict, List, Optional

import weaviate
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import settings


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


class KiaBrochureRAG:
    def __init__(self):
        self.client: Optional[weaviate.Client] = None
        self.embedding_model: Optional[SentenceTransformer] = None

    def _get_embedding_model(self) -> SentenceTransformer:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self.embedding_model

    def connect(self):
        if self.client is not None:
            return self.client

        if not settings.WEAVIATE_URL or not settings.WEAVIATE_API_KEY:
            raise RuntimeError(
                "WEAVIATE_URL and WEAVIATE_API_KEY must be set before querying the RAG database."
            )

        self.client = weaviate.Client(
            url=settings.WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=settings.WEAVIATE_API_KEY),
            additional_headers={"X-OpenAI-Api-Key": ""},
        )
        return self.client

    def ensure_collection(self):
        self.connect()
        if self.client is None:
            raise RuntimeError("Weaviate client is unavailable.")

        if self.client.schema.exists(settings.WEAVIATE_COLLECTION):
            return

        schema = {
            "classes": [
                {
                    "class": settings.WEAVIATE_COLLECTION,
                    "vectorizer": "none",
                    "properties": [
                        {"name": "title", "dataType": ["text"]},
                        {"name": "model", "dataType": ["text"]},
                        {"name": "year", "dataType": ["int"]},
                        {"name": "content", "dataType": ["text"]},
                        {"name": "source", "dataType": ["text"]},
                        {"name": "chunk_index", "dataType": ["int"]},
                    ],
                }
            ]
        }
        self.client.schema.create(schema)

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
        if not text:
            return []
        text = " ".join(text.split())
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    def _seed_documents(self) -> List[Dict[str, Any]]:
        brochures = [
            {
                "title": "Kia Sportage Brochure",
                "model": "Sportage",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Sportage is a compact SUV with a bold design, roomy interior, and advanced driver assistance systems. It offers efficient hybrid powertrains, a large digital cockpit, and flexible cargo space for everyday use.",
            },
            {
                "title": "Kia EV6 Brochure",
                "model": "EV6",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia EV6 is a high-performance all-electric crossover. It delivers rapid acceleration, fast charging, long-range battery options, and a premium infotainment experience designed for modern drivers.",
            },
            {
                "title": "Kia Sorento Brochure",
                "model": "Sorento",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Sorento is a midsize SUV focused on family comfort and versatility. It offers three rows of seating, flexible cargo space, available hybrid efficiency, and driver assistance features for long trips.",
            },
            {
                "title": "Kia Carnival Brochure",
                "model": "Carnival",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Carnival is a modern family vehicle built around comfort and convenience. It provides spacious seating, premium cabin technology, practical storage, and a smooth ride for everyday travel.",
            },
            {
                "title": "Kia Seltos Brochure",
                "model": "Seltos",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Seltos is a compact SUV with a youthful design, connected cabin technology, and versatile performance. It combines efficient engine options, comfortable handling, and smart interior practicality.",
            },
            {
                "title": "Kia Niro Brochure",
                "model": "Niro",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Niro is a compact crossover built around efficiency and everyday usability. Available in hybrid and electric variants, it focuses on lower running costs, comfort, and a practical design.",
            },
        ]
        return brochures

    def _pdf_documents(self, directory: str) -> List[Dict[str, Any]]:
        if not os.path.isdir(directory):
            return []

        docs: List[Dict[str, Any]] = []
        for file_name in sorted(os.listdir(directory)):
            if not file_name.lower().endswith(".pdf"):
                continue
            path = os.path.join(directory, file_name)
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not text.strip():
                continue
            docs.append(
                {
                    "title": os.path.splitext(file_name)[0],
                    "model": os.path.splitext(file_name)[0].replace("_", " "),
                    "year": 2025,
                    "source": path,
                    "content": text,
                }
            )
        return docs

    def _ingest_documents(self, documents: List[Dict[str, Any]]):
        self.ensure_collection()
        if self.client is None:
            raise RuntimeError("Weaviate client is unavailable.")

        self.client.batch.configure(batch_size=20)
        model = self._get_embedding_model()

        with self.client.batch as batch:
            for doc in documents:
                chunks = self._chunk_text(doc["content"])
                for chunk_index, chunk in enumerate(chunks):
                    vector = model.encode(chunk).tolist()
                    batch.add_data_object(
                        data_object={
                            "title": doc["title"],
                            "model": doc["model"],
                            "year": int(doc.get("year", 2025)),
                            "content": chunk,
                            "source": doc.get("source", "seed"),
                            "chunk_index": chunk_index,
                        },
                        class_name=settings.WEAVIATE_COLLECTION,
                        vector=vector,
                    )

    def ingest_brochures(self, directory: str | None = None):
        docs: List[Dict[str, Any]] = []
        if directory:
            docs = self._pdf_documents(directory)
        if not docs:
            docs = self._seed_documents()
        self._ingest_documents(docs)
        return len(docs)

    def query(self, question: str, top_k: int = 4):
        self.connect()
        self.ensure_collection()
        if self.client is None:
            raise RuntimeError("Weaviate client is unavailable.")

        vector = self._get_embedding_model().encode(question).tolist()
        response = (
            self.client.query.get(
                settings.WEAVIATE_COLLECTION,
                ["title", "model", "year", "content", "source", "chunk_index"],
            )
            .with_near_vector({"vector": vector})
            .with_limit(top_k)
            .do()
        )

        matches = response.get("data", {}).get("Get", {}).get(settings.WEAVIATE_COLLECTION, [])
        return matches

    def build_answer_from_context(self, question: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return (
                "I could not find any matching Kia brochure content. "
                "Please ingest brochure files or try a different question."
            )

        context = "\n\n".join(item.get("content", "") for item in results if item.get("content"))
        question_lower = question.lower()
        matched = []
        for item in results:
            content = item.get("content", "")
            haystack = content.lower()
            if any(token in haystack for token in question_lower.split() if len(token) > 2):
                matched.append(item)

        if matched:
            best = matched[0]
            answer = best.get("content", "")
            title = best.get("title", "Kia brochure")
            model = best.get("model", "Kia")
            return f"Based on the {model} brochure ({title}), the key detail is: {answer[:500]}"

        top = results[0]
        title = top.get("title", "Kia brochure")
        model = top.get("model", "Kia")
        summary = context[:700]
        return f"From the {model} materials ({title}), the brochure says: {summary}"


app = FastAPI(title="Kia Brochure RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    app.state.rag = KiaBrochureRAG()


@app.get("/", response_class=HTMLResponse)
async def home_page():
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Kia Easy Care Assistant</title>
            <style>
                :root {
                    --kia-red: #c4172a;
                    --kia-red-dark: #8f0d1d;
                    --bg: #0d1117;
                    --panel: rgba(18, 24, 32, 0.9);
                    --panel-soft: rgba(30, 39, 48, 0.9);
                    --text: #edf2f7;
                    --muted: #b7c2d1;
                    --line: rgba(255, 255, 255, 0.08);
                    --bubble-user: #c4172a;
                    --bubble-bot: #161d27;
                    --shadow: 0 28px 56px rgba(0, 0, 0, 0.35);
                }

                * { box-sizing: border-box; }

                body {
                    margin: 0;
                    min-height: 100vh;
                    font-family: Inter, "Segoe UI", sans-serif;
                    background:
                        radial-gradient(circle at top, rgba(196, 23, 42, 0.25), transparent 28%),
                        linear-gradient(135deg, #04070b 0%, #0d1117 42%, #101820 100%);
                    color: var(--text);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 24px;
                }

                .chat-shell {
                    width: min(100%, 980px);
                    background: var(--panel);
                    border: 1px solid var(--line);
                    border-radius: 26px;
                    box-shadow: var(--shadow);
                    overflow: hidden;
                    backdrop-filter: blur(14px);
                }

                .topbar {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 16px;
                    padding: 20px 24px;
                    background: rgba(255,255,255,0.02);
                    border-bottom: 1px solid var(--line);
                }

                .brand {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    font-weight: 700;
                    letter-spacing: 0.03em;
                }

                .brand-mark {
                    width: 40px;
                    height: 40px;
                    border-radius: 12px;
                    display: grid;
                    place-items: center;
                    background: linear-gradient(135deg, var(--kia-red) 0%, var(--kia-red-dark) 100%);
                    font-size: 1.1rem;
                    box-shadow: 0 12px 30px rgba(196, 23, 42, 0.45);
                }

                .status {
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 12px;
                    border-radius: 999px;
                    background: rgba(58, 197, 132, 0.12);
                    border: 1px solid rgba(58, 197, 132, 0.35);
                    color: #99f1bd;
                    font-size: 0.8rem;
                }

                .status::before {
                    content: "";
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #3bc57f;
                    display: block;
                    box-shadow: 0 0 12px rgba(59, 197, 127, 0.8);
                }

                .messages {
                    max-height: 520px;
                    overflow-y: auto;
                    padding: 26px 22px 18px;
                    background: rgba(10, 14, 18, 0.8);
                }

                .message {
                    display: flex;
                    margin-bottom: 18px;
                    animation: fadeIn 0.2s ease;
                }

                .message.user { justify-content: flex-end; }
                .message.bot { justify-content: flex-start; }

                .bubble {
                    max-width: min(76%, 620px);
                    padding: 14px 16px;
                    border-radius: 18px;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    word-break: break-word;
                }

                .message.user .bubble {
                    background: linear-gradient(135deg, var(--kia-red) 0%, #d83a4f 100%);
                    color: white;
                    border-bottom-right-radius: 6px;
                }

                .message.bot .bubble {
                    background: var(--bubble-bot);
                    border: 1px solid var(--line);
                    color: var(--text);
                    border-bottom-left-radius: 6px;
                }

                .composer {
                    display: flex;
                    gap: 14px;
                    align-items: flex-end;
                    padding: 20px 22px 24px;
                    border-top: 1px solid var(--line);
                    background: rgba(255,255,255,0.02);
                }

                .input-wrap {
                    flex: 1;
                    background: rgba(255,255,255,0.03);
                    border: 1px solid var(--line);
                    border-radius: 18px;
                    padding: 12px 14px;
                }

                textarea {
                    width: 100%;
                    resize: none;
                    border: none;
                    outline: none;
                    background: transparent;
                    color: var(--text);
                    font: inherit;
                    min-height: 56px;
                    max-height: 180px;
                }

                textarea::placeholder { color: var(--muted); }

                button {
                    border: none;
                    border-radius: 16px;
                    background: linear-gradient(135deg, var(--kia-red) 0%, #d9464e 100%);
                    color: white;
                    font: inherit;
                    font-weight: 700;
                    padding: 16px 22px;
                    cursor: pointer;
                    transition: transform 0.15s ease, opacity 0.2s ease;
                }

                button:hover:not(:disabled) { transform: translateY(-1px); }
                button:disabled { opacity: 0.7; cursor: wait; }

                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(4px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                @media (max-width: 620px) {
                    .topbar { padding: 16px 18px; }
                    .brand { font-size: 0.9rem; }
                    .messages { padding: 18px 16px 12px; }
                    .composer { padding: 16px; }
                    .bubble { max-width: 85%; }
                    .composer { flex-direction: column; }
                    button { width: 100%; }
                }
            </style>
        </head>
        <body>
            <div class="chat-shell">
                <div class="topbar">
                    <div class="brand">
                        <div class="brand-mark">K</div>
                        <span>Kia Easy Care</span>
                    </div>
                    <div class="status">Online</div>
                </div>

                <div id="chatMessages" class="messages">
                    <div class="message bot">
                        <div class="bubble">Hi! I'm Kia Easy Care. Ask me about Kia models, features, comfort, efficiency, and vehicle guidance.</div>
                    </div>
                </div>

                <form id="chatForm" class="composer">
                    <div class="input-wrap">
                        <textarea id="questionInput" placeholder="Ask about Kia vehicles, technology, or care support..." aria-label="Ask Kia Easy Care"></textarea>
                    </div>
                    <button type="submit">Ask</button>
                </form>
            </div>

            <script>
                const form = document.getElementById('chatForm');
                const input = document.getElementById('questionInput');
                const messages = document.getElementById('chatMessages');

                function addMessage(role, text) {
                    const row = document.createElement('div');
                    row.className = 'message ' + role;

                    const bubble = document.createElement('div');
                    bubble.className = 'bubble';
                    bubble.textContent = text;

                    row.appendChild(bubble);
                    messages.appendChild(row);
                    messages.scrollTop = messages.scrollHeight;
                }

                form.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const question = input.value.trim();
                    if (!question) return;

                    addMessage('user', question);
                    input.value = '';
                    input.focus();

                    const submitButton = form.querySelector('button');
                    submitButton.disabled = true;
                    submitButton.textContent = 'Thinking...';

                    const pending = document.createElement('div');
                    pending.className = 'message bot';
                    const pendingBubble = document.createElement('div');
                    pendingBubble.className = 'bubble';
                    pendingBubble.textContent = 'Thinking...';
                    pending.appendChild(pendingBubble);
                    messages.appendChild(pending);

                    try {
                        const response = await fetch('/ask', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ question, top_k: 4 })
                        });

                        const data = await response.json();
                        if (!response.ok) {
                            throw new Error(data.detail || 'Something went wrong.');
                        }

                        const answer = data.answer || 'I could not find a relevant answer.';
                        pending.remove();
                        addMessage('bot', answer);
                    } catch (error) {
                        pending.remove();
                        addMessage('bot', 'I am unable to answer that right now. Please try a different question or check the service status.');
                    } finally {
                        submitButton.disabled = false;
                        submitButton.textContent = 'Ask';
                    }
                });
            </script>
        </body>
        </html>
        """
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "kia-brochure-rag"}


@app.post("/ask")
async def ask_question(payload: AskRequest):
    rag = app.state.rag

    try:
        if settings.WEAVIATE_URL and settings.WEAVIATE_API_KEY:
            results = rag.query(payload.question, payload.top_k)
        else:
            directory = settings.DATA_DIR
            if os.path.isdir(directory):
                rag.ingest_brochures(directory)
            else:
                rag.ingest_brochures()
            results = rag.query(payload.question, payload.top_k)

        answer = rag.build_answer_from_context(payload.question, results)
        return {
            "question": payload.question,
            "answer": answer,
            "results": results,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for runtime issues
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
