import os
import time
from dotenv import load_dotenv
load_dotenv()

from clients import get_qdrant_client, get_openai_client
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType
from rag import generate_response

q_client = get_qdrant_client()
openai_client = get_openai_client()

collection_name = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")

# Step 0: Ensure Qdrant has a keyword index on the 'source' field so we can filter by filename
try:
    q_client.create_payload_index(
        collection_name=collection_name,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("[Step 0] Keyword payload index ensured on 'source' field in Qdrant.")
except Exception as e:
    print("[Step 0] Payload index ready.")

query_hindi = "यदि कोई कमर्शियल वाहन बिना परमिट के चलाया जा रहा है, तो कौन-सी धारा लागू होगी?"

print("====================================================================")
print("     TESTING QUERY USING ONLY THE 2019 AMENDMENT ACT FILTER         ")
print("====================================================================")
print(f"User Query (Hindi):\n  {query_hindi}\n")

# Step 1: Translate to English search keywords
t0 = time.perf_counter()
trans_res = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Translate the traffic law query into English legal keywords."},
        {"role": "user", "content": query_hindi}
    ],
    temperature=0,
    max_tokens=30
)
search_query_en = trans_res.choices[0].message.content.strip()
t1 = time.perf_counter()
print(f"[Step 1] English Search Query: '{search_query_en}'\n")

# Step 2: Embed query and search Qdrant with FILTER for Motor Vehicles (Amendment) Act, 2019.pdf only!
emb = openai_client.embeddings.create(
    model="text-embedding-3-large",
    input=[search_query_en]
).data[0].embedding
t2 = time.perf_counter()

# Filter strictly to the 2019 Amendment Act document!
target_source = "data/Motor Vehicles (Amendment) Act, 2019.pdf"
query_filter = Filter(
    must=[
        FieldCondition(
            key="source",
            match=MatchValue(value=target_source)
        )
    ]
)

results = q_client.query_points(
    collection_name=collection_name,
    query=emb,
    query_filter=query_filter,
    limit=5,
    with_payload=True
).points
t3 = time.perf_counter()

print(f"[Step 2] Found {len(results)} matches in '{target_source}'")
context_chunks = []
for idx, r in enumerate(results, 1):
    chunk_text = r.payload.get("text", "")
    context_chunks.append(chunk_text)
    print(f"  -> Match #{idx} | Score: {r.score:.4f} | Snippet: {chunk_text[:100].replace(chr(10), ' ')}")

context_string = "\n".join(context_chunks[:3])
print(f"\nTotal Retrieval Time: {(t3-t1)*1000:.1f} ms\n")

# Step 3: Generate structured Hindi legal answer
t4 = time.perf_counter()
answer = generate_response(
    query=query_hindi,
    context=context_string,
    language="hi"
)
t5 = time.perf_counter()

print("[Step 3] AI Answer Generated:")
print("====================== FINAL HINDI LEGAL ANSWER ======================")
print(answer)
print("======================================================================")
print(f"TOTAL END-TO-END LATENCY: {(t5-t0):.2f} seconds")
