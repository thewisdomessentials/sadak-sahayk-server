import os
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

from clients import get_qdrant_client

q_client = get_qdrant_client()
collection_name = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")

print(f"Connecting to Qdrant Cloud to analyze collection: '{collection_name}'...\n")

try:
    collection_info = q_client.get_collection(collection_name)
    print(f"📊 Collection Overview:")
    print(f"  -> Total vectors indexed: {collection_info.points_count}")
    
    # Scroll through points to collect unique sources and count chunks per source
    source_counts = Counter()
    
    offset = None
    while True:
        results, next_page_offset = q_client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=["source"],
            with_vectors=False
        )
        
        for point in results:
            src = point.payload.get("source", "Unknown Source")
            source_counts[src] += 1
            
        if next_page_offset is None:
            break
        offset = next_page_offset

    print("\n📂 Documents currently fed into Qdrant (by chunk count):")
    print("-" * 70)
    for src, count in source_counts.most_common():
        print(f"  • {src} (Chunks: {count})")
    print("-" * 70)
    print(f"\n✅ Total unique documents found in Qdrant: {len(source_counts)}")
    
except Exception as e:
    print(f"Error accessing Qdrant: {e}")
