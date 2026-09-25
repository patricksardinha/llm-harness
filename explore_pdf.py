from pypdf import PdfReader
import pymupdf

path = "data/World_Triathlon_Competition_Rules_2026.pdf"


def separator(page_number: int) -> str:
    return f"\n\n========== PAGE {page_number} ==========\n\n"


def separator_table() -> str:
    return f"\n\n========== TABLE ==========\n\n"

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
        f.write(separator(page.number + 1))
        tabs = page.find_tables()
        print(f"{len(tabs.tables)} tableau(x) trouvé(s) sur la page {page.number + 1}.")

        for tab in tabs:
            f.write(separator_table())
            data = tab.extract()
            print(f"len(data): {len(data)} | len(data[0]): {len(data[0])}")
            for line in data:
                l = " | ".join(cell if cell is not None else "" for cell in line)
                f.write(f"{l}\n")

print(f"pymupdf : {doc.page_count} pages -> data/extrait_pymupdf.txt")
doc.close()
