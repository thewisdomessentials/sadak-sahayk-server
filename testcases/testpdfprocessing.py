import fitz  # PyMuPDF

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        text += page_text + "\n"

    return text

# Chunk the text
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_legal_document(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n", ". ", " ", ""]
    )

    chunks = []
    current = ""

    for para in paragraphs:

        # If adding paragraph stays under limit
        if len(current) + len(para) < 800:
            current += "\n\n" + para

        else:
            # save previous chunk
            if current:
                chunks.append(current.strip())

            # Paragraph itself too large
            if len(para) > 800:
                chunks.extend(splitter.split_text(para))
                current = ""
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return chunks


if __name__ == "__main__":
    pdf_path = "CG Motor Vehicle Taxation Act and Rule 1991 (2024 Print)_organized_ocr.pdf"

    print("Loading PDF...")
    text = extract_text_from_pdf(pdf_path)

    print(f"\nTotal characters extracted: {len(text)}")
    print("\n===== FIRST 2000 CHARACTERS =====\n")
    print(text[:2000])

    chunks = chunk_legal_document(text)

    print(f"\nTotal chunks created: {len(chunks)}")

    # Print first few chunks
    for i, chunk in enumerate(chunks[5:9]):
        print(f"\n{'='*70}")
        print(f"CHUNK {i+1}")
        print(f"Length: {len(chunk)} characters")
        print(f"{'='*70}")
        print(chunk)