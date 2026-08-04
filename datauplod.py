import os
import uuid
import fitz  # PyMuPDF
from openai import OpenAI
from langdetect import detect
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

BATCH_EMBED_SIZE = 100
BATCH_UPSERT_SIZE = 20

# ----------------------------
# Helper to batch lists
# ----------------------------
def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


# ----------------------------
# Extract text from PDF
# ----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    return text


# ----------------------------
# Smart Legal Chunking
# ----------------------------
def chunk_legal_document(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []
    current = ""

    for para in paragraphs:

        # Add paragraph if it still fits
        if len(current) + len(para) < 800:
            if current:
                current += "\n\n"
            current += para

        else:

            # Save previous chunk
            if current:
                chunks.append(current.strip())

            # Very long paragraph
            if len(para) > 800:
                chunks.extend(splitter.split_text(para))
                current = ""
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return chunks


# ----------------------------
# OpenAI Embeddings
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts):
    embeddings = []

    for text_batch in batched(texts, BATCH_EMBED_SIZE):
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text_batch
        )

        embeddings.extend(item.embedding for item in response.data)

    return embeddings


# ----------------------------
# Qdrant
# ----------------------------
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
    )


# ----------------------------
# Upload to Qdrant
# ----------------------------
def append_to_qdrant(client, chunks, embeddings, source_file):
    collection_name = os.getenv("QDRANT_COLLECTION")

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        try:
            lang = detect(chunk)
        except Exception:
            lang = "unknown"

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": source_file,
                    "language": lang,
                },
            )
        )

    for index, batch in enumerate(
        batched(points, BATCH_UPSERT_SIZE),
        start=1,
    ):
        print(
            f"Uploading batch {index} "
            f"({len(batch)} chunks)..."
        )

        client.upsert(
            collection_name=collection_name,
            points=batch,
        )


# ----------------------------
# Main Ingestion Flow
# ----------------------------
def ingest_local_pdf(pdf_path):

    qdrant_client = get_qdrant_client()

    print(f"Processing: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_legal_document(text)
    chunks = chunks[5:]
    print(f"Created {len(chunks)} smart chunks.")

    # Optional preview
    print("\nFirst Chunk Preview:\n")
    print(chunks[0][:888])

    print("\nGenerating embeddings...")
    embeddings = get_embeddings(chunks)

    append_to_qdrant(
        qdrant_client,
        chunks,
        embeddings,
        pdf_path,
    )

    print("✅ Upload Complete!")


if __name__ == "__main__":

    LOCAL_PDF_NAME = "CG Motor Vehicle Taxation Act and Rule 1991 (2024 Print)_organized_ocr.pdf"

    ingest_local_pdf(LOCAL_PDF_NAME)