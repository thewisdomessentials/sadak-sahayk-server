import os
import re
import uuid
import fitz  # PyMuPDF
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

load_dotenv()

BATCH_EMBED_SIZE = 100
BATCH_UPSERT_SIZE = 20

def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc, 1):
        text += page.get_text("text") + "\n"
    print(f"Extracted text from {len(doc)} pages.")
    return text

def chunk_legal_document(text, max_chars=800, overlap=100):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            if current:
                current += "\n\n"
            current += para
        else:
            if current:
                chunks.append(current.strip())
            if len(para) > max_chars:
                start = 0
                while start < len(para):
                    end = start + max_chars
                    chunk_slice = para[start:end].strip()
                    if chunk_slice:
                        chunks.append(chunk_slice)
                    start += max_chars - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current.strip())
    return chunks

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts):
    embeddings = []
    for text_batch in batched(texts, BATCH_EMBED_SIZE):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text_batch
            )
            embeddings.extend(item.embedding for item in response.data)
            import time
            time.sleep(1)
        except Exception as e:
            print(f"    ⚠️ API/Network error hit: {e}. Sleeping for 10 seconds before retrying...")
            import time
            time.sleep(10)
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text_batch
            )
            embeddings.extend(item.embedding for item in response.data)
    return embeddings

def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
    )

def append_to_qdrant(qdrant_client, chunks, embeddings, source_file):
    collection_name = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": source_file,
                    "language": "en",  # Unified search assumes en
                },
            )
        )
    for index, batch in enumerate(batched(points, BATCH_UPSERT_SIZE), start=1):
        print(f"  -> Uploading batch {index} ({len(batch)} chunks)...")
        for attempt in range(3):
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
                break
            except Exception as e:
                print(f"    ⚠️ Qdrant connection error hit: {e}. Retrying {attempt+1}/3 in 5s...")
                import time
                time.sleep(5)

def main():
    qdrant_client = get_qdrant_client()

    pdf_path = os.path.join("data", "Traffic_Law_RAG_Failure_KnowledgeBase_Current_Law_2026.pdf")
    print(f"\nStep 1: Extracting text from '{pdf_path}'...")
    text = extract_text_from_pdf(pdf_path)

    print("\nStep 2: Chunking document...")
    chunks = chunk_legal_document(text)
    print(f"  -> Created {len(chunks)} legal chunks.")
    print("  -> First chunk preview:\n     " + repr(chunks[0][:150]))

    print("\nStep 3: Generating 3072-dimension embeddings (text-embedding-3-large)...")
    embeddings = get_embeddings(chunks)
    print(f"  -> Generated {len(embeddings)} embeddings.")

    print("\nStep 4: Upserting into Qdrant collection (NO existing documents are deleted!)...")
    append_to_qdrant(qdrant_client, chunks, embeddings, "data/Traffic_Law_RAG_Failure_KnowledgeBase_Current_Law_2026.pdf")

    print("\nINGESTION COMPLETE! Hindi Failures PDF indexed alongside existing documents!")

if __name__ == "__main__":
    main()
