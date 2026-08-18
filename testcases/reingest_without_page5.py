import os
import uuid
import fitz  # PyMuPDF
from openai import OpenAI
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from ingest_batch import chunk_legal_document, batched, BATCH_EMBED_SIZE, BATCH_UPSERT_SIZE

load_dotenv()

def main():
    q_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
    )
    collection_name = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    source_name = "data/TWE_Traffic_Law_AI_Knowledge_Base_2026_CLEAN_CURRENT_LAW.pdf"
    
    print(f"Deleting existing points for {source_name}...")
    q_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_name)
                    )
                ]
            )
        )
    )
    
    print(f"Extracting text from {source_name} (excluding page 5)...")
    doc = fitz.open(source_name)
    text = ""
    # Only read up to page 4 (index 3) out of 5 total pages.
    # The user wants to remove page 5, which is index 4.
    for i in range(len(doc)):
        if i == 4:
            continue
        text += doc[i].get_text("text") + "\n"
        
    print(f"Extracted text from {len(doc) - 1} pages.")
    
    chunks = chunk_legal_document(text)
    print(f"Created {len(chunks)} chunks.")
    
    if not chunks:
        print("No chunks to process.")
        return

    print("Generating Embeddings...")
    embeddings = []
    for text_batch in batched(chunks, BATCH_EMBED_SIZE):
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text_batch
        )
        embeddings.extend(item.embedding for item in response.data)
        import time
        time.sleep(1)

    print("Upserting to Qdrant...")
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": source_name,
                    "language": "en"
                },
            )
        )
        
    for batch in batched(points, BATCH_UPSERT_SIZE):
        q_client.upsert(
            collection_name=collection_name,
            points=batch,
        )
    print("Upload Complete.")

if __name__ == "__main__":
    main()
