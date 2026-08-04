import fitz

doc = fitz.open("data/MV ACT_Hindi.pdf")
print(f"Total pages in MV ACT_Hindi.pdf: {len(doc)}")

text_page_10 = doc[10].get_text("text")
print("--- Page 10 text sample (first 400 chars) ---")
print(text_page_10[:400])
