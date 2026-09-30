
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
    with open("data/extract_cleaned.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(list_pages):
            f.write(separator(i + 1))
            f.write(clean_page(page))


if __name__ == "__main__":
    parse_pdf(path)