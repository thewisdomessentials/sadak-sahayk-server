from qdrant_client import QdrantClient
from openai import OpenAI
from functools import lru_cache

try:
    from .config import get_secret
except ImportError:
    from config import get_secret

@lru_cache()
def get_qdrant_client():
    return QdrantClient(
        url=get_secret("qdrant-url"),
        api_key=get_secret("qdrant-api-key")
    )

@lru_cache()
def get_openai_client():
    return OpenAI(
        api_key=get_secret("openai-api-key")
    )
