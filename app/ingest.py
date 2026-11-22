import io
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_WORDS


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    raw = data.decode("utf-8", errors="replace")
    if name.endswith((".html", ".htm")) or raw.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return soup.get_text("\n")
    return strip_gutenberg(raw)


def strip_gutenberg(text: str) -> str:
    """Drop Project Gutenberg licence boilerplate so it doesn't pollute retrieval."""
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*", text, re.I)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", text, re.I)
    if start:
        text = text[start.end():]
        end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", text, re.I)
    if end:
        text = text[: end.start()]
    return text


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(words), step):
        piece = words[i : i + size]
        if len(piece) < 40 and out:  # tiny tail: already covered by the previous overlap
            break
        out.append(" ".join(piece))
        if i + size >= len(words):
            break
    return out
