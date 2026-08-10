import os
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()
# 1. Connect to your active cloud cluster
# (Hardcode these strings if your environment variables aren't loaded in this terminal)
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")

if not qdrant_url or not qdrant_api_key:
    print("❌ Error: Qdrant credentials missing. Please set env variables or hardcode them.")
    exit(1)

qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# 2. Initialize OpenAI for generating the query vector
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("❌ Error: OPENAI_API_KEY missing.")
    exit(1)

openai_client = OpenAI(api_key=openai_api_key)

# 3. Define a test query based on your new data
# (Change this text to match a specific keyword/rule inside your newly added text file!)
TEST_QUERY = "what is the fine for not giving ambulace way priority?"

print(f"🔍 Step 1: Generating 3072-dimension embedding for query: '{TEST_QUERY}'")

# Generate embedding matching the exact model used for ingestion
response = openai_client.embeddings.create(
    model="text-embedding-3-large",
    input=[TEST_QUERY],
    dimensions=3072
)
query_vector = response.data[0].embedding

print("📡 Step 2: Querying Qdrant Cloud Cluster...")

# Search the collection
search_results = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3  # Retrieve the top 3 closest matches
).points

print("\n=================== RETRIEVAL RESULTS ===================")
if not search_results:
    print("⚠️ No matching documents found. Is the collection name correct?")
else:
    for i, point in enumerate(search_results, start=1):
        payload = point.payload
        score = point.score  # Similarity score (closer to 1.0 is better)
        
        print(f"\n[Match #{i}] | Similarity Score: {score:.4f}")
        print(f"Source Document: {payload.get('source', 'Unknown')}")
        print(f"Language: {payload.get('language', 'Unknown')}")
        print("-" * 50)
        print(payload.get('text', 'No text payload found!'))
        print("=" * 50)

print("\n🚀 Test retrieval complete. Check your Qdrant Dashboard metrics in a few minutes to watch the request register!")