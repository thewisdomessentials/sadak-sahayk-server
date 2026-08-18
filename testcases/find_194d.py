import fitz

doc = fitz.open("data/Motor Vehicles (Amendment) Act, 2019.pdf")
print("Searching for '194D' in 2019 Amendment Act...\n")

found = False
for page_num, page in enumerate(doc, 1):
    text = page.get_text("text")
    if "194D" in text:
        found = True
        print(f"--- Found on Page {page_num} ---")
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "194D" in line:
                # Print context around the match
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                print("\n".join(lines[start:end]))
                print("-" * 40)

if not found:
    print("Could not find '194D' in the PDF!")
