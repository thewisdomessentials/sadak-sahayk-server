import os
import uuid
import fitz  # PyMuPDF
from openai import OpenAI, RateLimitError
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
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        print(f"    -> Extracted text from {len(doc)} pages.")
        return text
    except Exception as e:
        print(f"    -> ERROR reading {pdf_path}: {e}")
        return ""

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

def get_existing_sources(q_client, collection_name):
    print("🔍 Querying Qdrant for existing indexed documents...")
    existing_sources = set()
    try:
        offset = None
        while True:
            results, next_page_offset = q_client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for point in results:
                src = point.payload.get("source")
                if src:
                    existing_sources.add(src)
            if next_page_offset is None:
                break
            offset = next_page_offset
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
    return existing_sources

def main():
    q_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
    )
    collection_name = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    existing_sources = get_existing_sources(q_client, collection_name)
    print(f"✅ Found {len(existing_sources)} unique documents already in Qdrant.\n")

    directories_to_scan = ["data", "data/AIS", "data/BIS", "data/Notification Morth", "data/Gazzette", "data/Morth Circularand Circular", "data/AIS (400 pdf)"]
    files_to_process = []

    for directory in directories_to_scan:
        if not os.path.exists(directory):
            continue
        for filename in os.listdir(directory):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                # Normalize path for Qdrant storage (use forward slashes)
                qdrant_source_name = file_path.replace("\\", "/")
                
                if qdrant_source_name in existing_sources:
                    print(f"⏭️  SKIPPING (Already Indexed): {qdrant_source_name}")
                else:
                    files_to_process.append((file_path, qdrant_source_name))

    if not files_to_process:
        print("\n🎉 All PDFs are already indexed! Nothing new to process.")
        return

    print(f"\n🚀 Found {len(files_to_process)} NEW documents to ingest!\n")

    for idx, (file_path, source_name) in enumerate(files_to_process, 1):
        print(f"[{idx}/{len(files_to_process)}] Processing: {source_name}")
        
        text = extract_text_from_pdf(file_path)
        if not text:
            continue
            
        chunks = chunk_legal_document(text)
        print(f"    -> Created {len(chunks)} chunks.")
        if not chunks:
            continue

        print("    -> Generating Embeddings (with rate limit protection)...")
        embeddings = []
        for text_batch in batched(chunks, BATCH_EMBED_SIZE):
            try:
                response = openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text_batch
                )
                embeddings.extend(item.embedding for item in response.data)
                # Sleep for 1 second between batches to respect OpenAI TPM limits
                import time
                time.sleep(1)
            except Exception as e:
                print(f"    ⚠️ API/Network error hit: {e}. Sleeping for 10 seconds before retrying...")
                import time
                time.sleep(10)
                # Retry once
                response = openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text_batch
                )
                embeddings.extend(item.embedding for item in response.data)

        print("    -> Upserting to Qdrant...")
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": source_name,
                        "language": "en" # Defaulting to en for unified search
                    },
                )
            )
            
        for batch in batched(points, BATCH_UPSERT_SIZE):
            for attempt in range(3):
                try:
                    q_client.upsert(
                        collection_name=collection_name,
                        points=batch,
                    )
                    break
                except Exception as e:
                    print(f"    ⚠️ Qdrant connection error hit: {e}. Retrying {attempt+1}/3 in 5s...")
                    import time
                    time.sleep(5)
        print("    ✅ Upload Complete.\n")

    print("🎉 BATCH INGESTION FINISHED!")

if __name__ == "__main__":
    main()
