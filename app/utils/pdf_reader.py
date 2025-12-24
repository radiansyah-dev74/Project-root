# app/utils/pdf_reader.py

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract plain text from a PDF file.
    Simple & deterministic (sufficient for case study).
    """
    reader = PdfReader(file_path)

    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    if not pages_text:
        raise ValueError("PDF contains no readable text")

    return "\n".join(pages_text)
