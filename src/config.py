from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google_genai:gemini-3.1-flash-lite"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
RETRIEVER_K = 6