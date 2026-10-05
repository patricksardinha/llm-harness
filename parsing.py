
from pypdf import PdfReader
import pymupdf
import re

REBUTS = [
    r"^World Triathlon Competition Rules$",
    r"^\d+ \w+ \d{4}$",      # "13 December 2025"
    r"^\d+/\d+$",            # "16/212"
    r"^BACK TO$",
    r"^CONTENTS$",
]

CORPS = slice(9, 133)
CHAPITRE = r"^(\d+)\.\s+([A-Z][A-Z\s:]+)$"      # "2. CONDUCT OF ATHLETES"
SECTION  = r"^(\d+\.\d+)\s+(.+?):?\s*$"         # "2.1 General Conduct:"

path = "data/World_Triathlon_Competition_Rules_2026.pdf"

def separator(page_number: int) -> str:
    return f"\n\n========== PAGE {page_number} ==========\n\n"


def extract_pages(path: str) -> list[str]:
    pages = []
    reader = PdfReader(path)
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def clean_page(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        nue = line.strip()
        if any(re.match(motif, nue) for motif in REBUTS):
            continue
        lines.append(line)
    return "\n".join(lines)


def parse_pdf(path: str) -> list[str]:
    list_pages = extract_pages(path)
    list_cleaned_pages = []
    with open("data/extract_cleaned.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(list_pages):
            c_page = clean_page(page)
            f.write(separator(i + 1))
            f.write(c_page)
            list_cleaned_pages.append(c_page)
    return list_cleaned_pages


def check_pages(pages: list[str]) -> list[tuple[str, str, float, str]]:
    infos = []
    for i, page in enumerate(pages, start=10):
        for line in page.split("\n"):
            nue = line.strip()
            m = re.match(SECTION, nue)
            if m:
                # no page, type (section, chapitre), no, titre
                infos.append((i, "section", m.group(1), m.group(2)))
    return infos



if __name__ == "__main__":
    pages = parse_pdf(path)
    infos = check_pages(pages[CORPS])
    for info in infos:
        print(info)
