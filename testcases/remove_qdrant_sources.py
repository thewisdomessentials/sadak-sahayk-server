import os
from qdrant_client import models
from clients import get_qdrant_client

def remove_from_qdrant():
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")
    
    sources_to_delete = [
        "data/MVA 1988 till 2025 may.pdf"
    ]
    
    for source in sources_to_delete:
        print(f"Deleting points from source: {source}")
        response = client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source)
                        )
                    ]
                )
            )
        )
        print(f"Delete response: {response}")

if __name__ == "__main__":
    remove_from_qdrant()
