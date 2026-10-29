import os
from typing import Any, Dict, List

import weaviate
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import settings


class KiaBrochureRAG:
    def __init__(self):
        self.client = None
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def connect(self):
        if self.client is not None:
            return self.client

        if not settings.WEAVIATE_URL or not settings.WEAVIATE_API_KEY:
            raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set")

        self.client = weaviate.Client(
            url=settings.WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=settings.WEAVIATE_API_KEY),
            additional_headers={"X-OpenAI-Api-Key": ""},
        )
        return self.client

    def ensure_collection(self):
        self.connect()
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
                "content": "The Kia Sportage is a compact SUV with a bold design, roomy interior, and advanced driver assistance systems. It offers efficient hybrid powertrains, a large digital cockpit, smart connectivity, and comfortable seating for families. The Sportage includes safety features such as lane-keeping assist, forward collision warning, adaptive cruise control, and parking sensors.",
            },
            {
                "title": "Kia EV6 Brochure",
                "model": "EV6",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia EV6 is a high-performance all-electric crossover. It delivers sharp acceleration, rapid charging capability, a spacious cabin, and a premium infotainment setup. Range depends on trim and driving conditions, but the EV6 is designed for everyday commuting and longer highway trips with efficient battery management.",
            },
            {
                "title": "Kia Sorento Brochure",
                "model": "Sorento",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Sorento is a midsize SUV built for families and road trips. It offers three rows of seating, ample cargo space, versatile interior layouts, and available hybrid powertrain options. Safety features include blind-spot monitoring, lane centering, rear occupant alert, and intelligent cruise control.",
            },
            {
                "title": "Kia Carnival Brochure",
                "model": "Carnival",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Carnival is a modern MPV focused on comfort, convenience, and family utility. It offers flexible seating, generous cabin space, premium comfort features, and large infotainment screens. The Carnival includes practical storage, a smooth ride quality, and smart safety systems designed around family travel.",
            },
            {
                "title": "Kia Seltos Brochure",
                "model": "Seltos",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Seltos is a compact SUV with a youthful design, a technology-rich cabin, and versatile performance. It offers efficient engine choices, a comfortable ride, good visibility, and available safety technology. The Seltos is suitable for city driving and weekend adventures with practical storage and a compact footprint.",
            },
            {
                "title": "Kia Niro Brochure",
                "model": "Niro",
                "year": 2025,
                "source": "Seed data",
                "content": "The Kia Niro is a compact crossover designed around efficiency and everyday usability. Available in hybrid and electric variants, it emphasizes low running costs, smooth city driving, and a comfortable cabin. The Niro offers modern connectivity, useful safety features, and a practical interior for daily life.",
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
        self.client.batch.configure(batch_size=20)

        with self.client.batch as batch:
            for idx, doc in enumerate(documents):
                chunks = self._chunk_text(doc["content"])
                for chunk_index, chunk in enumerate(chunks):
                    vector = self.embedding_model.encode(chunk).tolist()
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
        vector = self.embedding_model.encode(question).tolist()
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
            return "I could not find any matching Kia brochure content. Please ingest brochure files or try a different question."

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
