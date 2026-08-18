import os
from dotenv import load_dotenv
load_dotenv()

from clients import get_qdrant_client, get_openai_client

q_client = get_qdrant_client()
openai_client = get_openai_client()

queries = [
    "driving commercial vehicle without permit penalty section 192A",
    "बिना परमिट वाहन चलाने पर जुर्माना धारा 192A"
]

for q in queries:
    print(f"\n================ QUERY: '{q}' ================")
    emb = openai_client.embeddings.create(model="text-embedding-3-large", input=[q]).data[0].embedding
    results = q_client.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents"),
        query=emb,
        limit=3,
        with_payload=True
    ).points
    for idx, r in enumerate(results, 1):
        print(f"--- [Match {idx}] (Score: {r.score:.4f}) | Source: {r.payload.get('source')} | Lang: {r.payload.get('language')} ---")
        print(r.payload.get("text", "")[:300])
