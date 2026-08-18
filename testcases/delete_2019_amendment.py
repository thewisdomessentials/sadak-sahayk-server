import os
from dotenv import load_dotenv
from qdrant_client.models import Filter, FieldCondition, MatchValue

from clients import get_qdrant_client

def main():
    load_dotenv()
    
    q_client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")
    
    source_to_delete = "data/Motor Vehicles (Amendment) Act, 2019.pdf"
    
    print(f"Deleting all chunks from Qdrant where source is exactly '{source_to_delete}'...")
    
    query_filter = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source_to_delete)
            )
        ]
    )
    
    response = q_client.delete(
        collection_name=collection_name,
        points_selector=query_filter,
    )
    
    print("\nDeletion Complete! Response from Qdrant:")
    print(response)

if __name__ == "__main__":
    main()
