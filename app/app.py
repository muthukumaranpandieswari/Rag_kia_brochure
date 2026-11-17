import os
import glob
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = FastAPI(
    title="Multimodal Car Brochure RAG API",
    description="Vector search and query generation on Kia Sorento brochure using NVIDIA Nemotron 3 Nano Omni."
)

# Extract API Key strictly from environment variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY environment variable is empty!")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="kia_sorento_brochure")

page_summaries = {
    1: "Kia Sorento Cover Page, logo, visual exterior shot, front design.",
    2: "Exterior styling, Crystal Cube LED Projection Headlamps, Tiger Face, panoramic sunroof, wheels.",
    3: "Interior cabin comfort, 7-seat configuration, premium leather seats, dashboard layout.",
    4: "Driver-centric cockpit, dual curved displays, infotainment, ambient lighting.",
    5: "Performance, engine options, Smartstream powertrain, AWD (All-Wheel Drive) features.",
    6: "Advanced Safety Features, Advanced Driver Assistance Systems (ADAS), Forward Collision-Avoidance.",
    7: "Smart connectivity, Kia Connect features, smartphone integration, voice controls.",
    8: "Trims, variant specifications, wheel configurations, color options, dimensions table.",
    9: "Back cover, Kia Care Network, warranty info (3-year unlimited km), customer contact details."
}

IMAGE_DIR = "/app/kia_pages"

def encode_image_to_base64(img_path, max_width=1200):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
    with Image.open(img_path) as img:
        if img.width > max_width:
            new_height = int(img.height * (max_width / img.width))
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG", quality=65)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.on_event("startup")
def index_pages():
    image_paths = sorted(glob.glob(f"{IMAGE_DIR}/*.png"))
    if not image_paths:
        print(f"Warning: No images found to index in '{IMAGE_DIR}'")
        return
    ids, documents, embeddings, metadatas = [], [], [], []
    for idx, path in enumerate(image_paths, start=1):
        summary = page_summaries.get(idx, f"Kia Sorento Brochure Page {idx}")
        vector = embedding_model.encode(summary).tolist()
        ids.append(f"page_{idx}")
        documents.append(summary)
        embeddings.append(vector)
        metadatas.append({
            "image_path": path,
            "page_number": idx,
            "car_model": "Kia Sorento",
            "year": 2026,
            "is_latest": True
        })
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(image_paths)} pages on startup.")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(payload: QueryRequest):
    query = payload.query
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured on the host system.")
    if collection.count() == 0:
        raise HTTPException(status_code=503, detail="Knowledge base has not been indexed yet.")
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=1, where={"is_latest": True})
    retrieved_meta = results['metadatas'][0][0]
    retrieved_doc = results['documents'][0][0]
    img_path = retrieved_meta['image_path']
    page_num = retrieved_meta['page_number']
    try:
        base64_image = encode_image_to_base64(img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process retrieved image: {e}")
    prompt = f"Based on the provided brochure image, please answer this question precisely: {query}"
    content_payload = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            messages=[{"role": "user", "content": content_payload}]
        )
        return {
            "query": query,
            "retrieved_page": page_num,
            "page_context": retrieved_doc,
            "response": response.choices[0].message.content
        }
    except Exception as e:
         raise HTTPException(status_code=502, detail=f"Model generation failed: {e}")

@app.get("/")
def root():
    return {"status": "running", "engine": "FastAPI", "RAG_collection_count": collection.count()}