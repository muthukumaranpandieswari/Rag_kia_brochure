import os
from typing import Any, Dict, List, Optional

import weaviate
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
                "content": "The Kia Sportage is a compact SUV with a bold design, roomy interior, and advanced driver assistance systems. It offers efficient hybrid powertrains, a large digital cockpit, and practical cargo space for daily use and family travel.",
            },
            {
                "title": "Kia EV6 Brochure",
                "model": "EV6",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia EV6 is a high-performance all-electric crossover. It delivers rapid acceleration, long-range capability, fast charging, and an advanced infotainment system designed for modern drivers.",
            },
            {
                "title": "Kia Sorento Brochure",
                "model": "Sorento",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Sorento is a midsize SUV built for families and road trips. It offers three rows of seating, flexible interior layouts, ample cargo space, and available hybrid powertrain options.",
            },
            {
                "title": "Kia Carnival Brochure",
                "model": "Carnival",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Carnival is a modern MPV focused on comfort, convenience, and family utility. It includes flexible seating, generous cabin space, premium comfort features, and a connected driver experience.",
            },
            {
                "title": "Kia Seltos Brochure",
                "model": "Seltos",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Seltos is a compact SUV with a youthful design, technology-rich cabin, and versatile performance. It offers efficient engine choices, a comfortable ride, and smart convenience features.",
            },
            {
                "title": "Kia Niro Brochure",
                "model": "Niro",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Niro is a compact crossover designed around efficiency and everyday usability. It is available in hybrid and electric variants and emphasizes low running costs, smooth driving, and practical design.",
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


@app.get("/")
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
