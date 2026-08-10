import os
import time
from dotenv import load_dotenv
load_dotenv()

from clients import get_qdrant_client, get_openai_client
from rag import generate_response

q_client = get_qdrant_client()
openai_client = get_openai_client()

english_query = "A driver of a private four-wheeler was found driving without wearing a seat belt during a routine traffic inspection. Which provision applies under the Motor Vehicles Act? Mention section, penalty, and source."

print("====================================================================")
print("             TESTING ENGLISH RAG (LATEST AMENDMENT ACT)             ")
print("====================================================================")
print(f"User Query:\n  {english_query}\n")

# Step 1: Vector Search against Qdrant (Searching ALL documents)
t0 = time.perf_counter()
emb = openai_client.embeddings.create(
    model="text-embedding-3-large",
    input=[english_query]
).data[0].embedding
t1 = time.perf_counter()

results = q_client.query_points(
    collection_name=os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents"),
    query=emb,
    limit=10,
    with_payload=True
).points
t2 = time.perf_counter()

# Filter for English chunks only (to avoid the corrupted Hindi PDF chunks)
en_chunks = []
for idx, r in enumerate(results, 1):
    lang = r.payload.get("language", "")
    src = r.payload.get("source", "")
    if "English" in src or "2019" in src or lang == "en":
        en_chunks.append(r.payload.get("text", ""))
        print(f"[Step 1] Retrieved Match #{len(en_chunks)} | Score: {r.score:.4f} | Source: {src}")

context_string = "\n".join(en_chunks[:6])
print(f"\nTotal Retrieval Time (Embed + Qdrant): {(t2-t0)*1000:.1f} ms\n")

print("--- Sample of Authority Legal Text Retrieved ---")
print(context_string[:450].replace("\n", " "))
print("------------------------------------------------\n")

# Step 2: Generate structured English legal answer using the English context
t3 = time.perf_counter()
answer = generate_response(
    query=english_query,
    context=context_string,
    language="en"
)
t4 = time.perf_counter()

print(f"[Step 2] AI Response Generated in {(t4-t3)*1000:.1f} ms")
print("====================== FINAL LEGAL ANSWER ======================")
print(answer)
print("================================================================")
print(f"TOTAL END-TO-END LATENCY: {(t4-t0):.2f} seconds")
