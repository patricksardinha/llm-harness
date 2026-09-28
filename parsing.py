
from pypdf import PdfReader
import pymupdf

path = "data/World_Triathlon_Competition_Rules_2026.pdf"


def separator(page_number: int) -> str:
    return f"\n\n========== PAGE {page_number} ==========\n\n"


def separator_table() -> str:
    return f"\n\n========== TABLE ==========\n\n"


def extract_pages(path: str) -> list[str]:      # PDF -> une chaîne par page
    pages = []
    reader = PdfReader(path)
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def clean_page(texte: str) -> str:              # retire en-tête et pied
    pass


def parse_pdf(path: str) -> list[str]:          # enchaîne les deux
    pass