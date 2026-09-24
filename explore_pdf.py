from pypdf import PdfReader
import pymupdf

path = "data/World_Triathlon_Competition_Rules_2026.pdf"


def separator(page_number: int) -> str:
    return f"\n\n========== PAGE {page_number} ==========\n\n"


# --- pypdf ---
reader = PdfReader(path)
with open("data/extrait_pypdf.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages, start=1):
        f.write(separator(i))
        f.write(page.extract_text() or "")

print(f"pypdf   : {len(reader.pages)} pages -> data/extrait_pypdf.txt")


# --- pymupdf ---
doc = pymupdf.open(path)
with open("data/extrait_pymupdf.txt", "w", encoding="utf-8") as f:
    for page in doc:
        f.write(separator(page.number + 1))
        f.write(page.get_text())

with open("data/extrait_table_pymupdf.txt", "w", encoding="utf-8") as f:
    for page in doc:
        tables = page.find_tables()
        for table in tables:
            f.write("------------ Normal ---------------")
            f.write(table.extract())
            f.write("------------ Pandas ---------------")
            f.write(table.to_pandas())
            f.write("------------ Markdown ---------------")
            f.write(table.to_markdown())

print(f"pymupdf : {doc.page_count} pages -> data/extrait_pymupdf.txt")
doc.close()
