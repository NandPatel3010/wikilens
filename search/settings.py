import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
EMBEDDER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CANDIDATE_DEPTH = 50
MAX_QUERY_LENGTH = 500

def database_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL in .env or in the service environment.")
    return url
