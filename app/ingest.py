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


SECTION_RE = re.compile(r"^\s*={3,}\s*(.+?)\s*={3,}\s*$", re.M)


def split_sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
    """Files can hold several works, each opened by a line like '==== Title ===='.
    Each chunk then knows which work it came from. Files without such lines are one section."""
    marks = list(SECTION_RE.finditer(text))
    if not marks:
        return [(fallback_title, text)]
    out = []
    if text[: marks[0].start()].strip():
        out.append((fallback_title, text[: marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1).strip(), text[m.end():end]))
    return [(t, body) for t, body in out if body.strip()]


def title_from_filename(filename: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename or "upload")
    return re.sub(r"[-_]+", " ", stem).strip() or "upload"


def embedding_text(person: str, section: str, chunk: str) -> str:
    """What gets embedded: the chunk plus who and what it is from. First-person text ("I was born...")
    rarely names its subject, so without this a question naming the person can't find it."""
    return f"Source: {section}\nAbout: {person}\n\n{chunk}"


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
