import time
import os
from dotenv import load_dotenv
load_dotenv()

from clients import get_qdrant_client, get_openai_client

q_client = get_qdrant_client()
openai_client = get_openai_client()

queries = [
    ("Hindi Query", "बिना परमिट कमर्शियल वाहन चलाने पर जुर्माना धारा 192A"),
    ("Hinglish Query", "bina permit commercial vehicle chalane par kya fine hai section 192A"),
    ("English Query", "driving commercial vehicle without permit penalty section 192A")
]

print("=== TESTING CROSS-LINGUAL SEARCH AGAINST ENGLISH DOCUMENTS IN QDRANT ===")
for label, q in queries:
    t0 = time.perf_counter()
    emb = openai_client.embeddings.create(model="text-embedding-3-large", input=[q]).data[0].embedding
    t1 = time.perf_counter()
    
    results = q_client.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents"),
        query=emb,
        limit=3,
        with_payload=True
    ).points
    t2 = time.perf_counter()
    
    print(f"\n[{label}] '{q}'")
    print(f"  -> Embedding Time: {(t1 - t0)*1000:.1f} ms | Qdrant Time: {(t2 - t1)*1000:.1f} ms")
    for idx, r in enumerate(results, 1):
        if r.payload.get("language") == "en":
            print(f"  -> [English Match #{idx}] Score: {r.score:.4f} | File: {r.payload.get('source')}")
            print("     Text snippet:", r.payload.get("text", "")[:120].replace("\n", " "))
            break
