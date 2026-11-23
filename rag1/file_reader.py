from pymupdf import open
import re
from typing import List
from pathlib import Path

def extract_paragraphs(pdf_path: str | Path) -> List[str]:
    """Extracts paragraphs from a PDF file."""

    doc = open(pdf_path)
    paragraphs = []

    for page in doc:
        blocks = page.get_text("blocks")

        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        for block in blocks:
            text = block[4]

            #replace line breaks with spaces
            text = re.sub(r"\r+", " ", text).strip()

            if text:
                paragraphs.append(text)
    return paragraphs

print(len(extract_paragraphs("data/dalai_lama_purpose_of_life_essay.pdf")))