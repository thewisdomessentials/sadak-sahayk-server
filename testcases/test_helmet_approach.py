import os
import time
from dotenv import load_dotenv
load_dotenv()

from clients import get_qdrant_client, get_openai_client
from rag import generate_response

q_client = get_qdrant_client()
openai_client = get_openai_client()

hindi_query = "यदि कोई व्यक्ति बिना हेलमेट के मोटरसाइकिल चलाते हुए पकड़ा जाता है, तो उस पर मोटर वाहन अधिनियम की कौन-सी धारा लागू होगी?"

print("====================================================================")
print("             TESTING ENGLISH-CORE + HINDI-OUTPUT RAG                ")
print("====================================================================")
print(f"Original User Query (Hindi):\n  {hindi_query}\n")

# Step 1: Quick English translation of search keywords (Stage 1 of approach)
t0 = time.perf_counter()
trans_res = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Translate the user's traffic law query into concise English search keywords for a legal statute database."},
        {"role": "user", "content": hindi_query}
    ],
    temperature=0,
    max_tokens=30
)
english_search_query = trans_res.choices[0].message.content.strip()
t1 = time.perf_counter()
print(f"[Step 1] Translated Search Query: '{english_search_query}' (Took {(t1-t0)*1000:.1f} ms)\n")

# Step 2: Vector Search against Qdrant English Documents
emb = openai_client.embeddings.create(
    model="text-embedding-3-large",
    input=[english_search_query]
).data[0].embedding
t2 = time.perf_counter()

results = q_client.query_points(
    collection_name=os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents"),
    query=emb,
    limit=5,
    with_payload=True
).points
t3 = time.perf_counter()

# Filter for English chunks only
en_chunks = []
for idx, r in enumerate(results, 1):
    lang = r.payload.get("language", "")
    src = r.payload.get("source", "")
    # Prefer English documents or chunks
    if "English" in src or lang == "en":
        en_chunks.append(r.payload.get("text", ""))
        print(f"[Step 2] Retrieved Match #{len(en_chunks)} | Score: {r.score:.4f} | Source: {src}")

context_string = "\n".join(en_chunks[:3])
print(f"Total Retrieval Time (Embed + Qdrant): {(t3-t1)*1000:.1f} ms\n")

print("--- Sample of Authority English Legal Text Retrieved ---")
print(context_string[:350].replace("\n", " "))
print("------------------------------------------------------\n")

# Step 3: Generate structured Hindi legal answer using the English context
t4 = time.perf_counter()
answer = generate_response(
    query=hindi_query,
    context=context_string,
    language="hi"
)
t5 = time.perf_counter()

print(f"[Step 3] AI Response Generated in {(t5-t4)*1000:.1f} ms")
print("====================== FINAL HINDI LEGAL ANSWER ======================")
print(answer)
print("======================================================================")
print(f"TOTAL END-TO-END LATENCY: {(t5-t0):.2f} seconds")
