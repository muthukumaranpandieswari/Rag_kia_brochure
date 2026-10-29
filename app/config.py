import os
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "").strip()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "").strip()
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "KiaBrochure")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_DIR = os.getenv("DATA_DIR", "./data/brochures")

if WEAVIATE_URL and not WEAVIATE_URL.startswith("http"):
    WEAVIATE_URL = f"https://{WEAVIATE_URL}"


class AppSettings:
    WEAVIATE_URL = WEAVIATE_URL
    WEAVIATE_API_KEY = WEAVIATE_API_KEY
    WEAVIATE_COLLECTION = WEAVIATE_COLLECTION
    EMBEDDING_MODEL = EMBEDDING_MODEL
    DATA_DIR = DATA_DIR


settings = AppSettings()
