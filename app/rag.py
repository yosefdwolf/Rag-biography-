import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sqlalchemy import insert, select, update

from . import llm
from .config import CHAT_MODEL, CHUNK_OVERLAP, CHUNK_WORDS, EMBED_MODEL, MIN_SIMILARITY, TOP_K
from .db import chunks, documents, dumps, engine, new_id, now, query_logs, runs

NO_CALL = {"tokens_in": None, "tokens_out": None, "latency_ms": None}


def log(collection_id, kind, question, retrieved, prompt, response, meta, llm_called, run_id=None, qnum=None):
    with engine.begin() as c:
        c.execute(insert(query_logs).values(
            run_id=run_id, collection_id=collection_id, kind=kind, qnum=qnum, question=question,
            retrieved=dumps(retrieved), prompt=prompt, response=response, model=CHAT_MODEL,
            llm_called=int(llm_called), created_at=now(), **meta,
        ))

NOT_FOUND = "NOT IN THE SOURCES"

QUESTIONS = [
    "Where and when was {person} born?",
    "Where were {person}'s parents from, and what did they do?",
    "What was {person}'s childhood like?",
    "Who were {person}'s siblings?",
    "What schooling or training did {person} have?",
    "What was {person}'s first job?",
    "An adventurous story from {person}'s life.",
    "A scary or dangerous moment {person} faced.",
    "A funny anecdote about {person}.",
    "A time {person} failed or made a serious mistake.",
    "The hardest thing {person} went through.",
    "Who did {person} marry, and what was that relationship like?",
    "What about {person}'s children?",
    "Who were {person}'s closest friends or mentors?",
    "Who were {person}'s rivals or enemies?",
    "What is {person} best known for?",
    "Something {person} made, invented or discovered.",
    "Where did {person} travel or live?",
    "A habit, hobby or quirk of {person}.",
    "How did {person} die, and how is {person} remembered?",
]

CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*)\]")

# ---------- retrieval ----------

_cache: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
_cache_lock = threading.Lock()


def invalidate(collection_id: str):
    with _cache_lock:
        _cache.pop(collection_id, None)


def _load(collection_id: str):
    with _cache_lock:
        if collection_id in _cache:
            return _cache[collection_id]
    with engine.connect() as c:
        rows = c.execute(
            select(chunks.c.id, chunks.c.text, chunks.c.embedding)
            .where(chunks.c.collection_id == collection_id)
            .order_by(chunks.c.id)
        ).all()
    ids = [r.id for r in rows]
    texts = [r.text for r in rows]
    mat = (
        np.vstack([np.frombuffer(r.embedding, dtype=np.float32) for r in rows])
        if rows else np.zeros((0, 1), dtype=np.float32)
    )
    with _cache_lock:
        _cache[collection_id] = (ids, texts, mat)
    return ids, texts, mat


def retrieve(collection_id: str, query: str, k: int = TOP_K):
    """Top-k chunks by cosine similarity, keeping only those above MIN_SIMILARITY."""
    ids, texts, mat = _load(collection_id)
    if not ids:
        return []
    q = llm.embed([query])[0]
    scores = mat @ q
    order = np.argsort(-scores)[:k]
    return [
        {"chunk_id": ids[i], "score": round(float(scores[i]), 4), "text": texts[i]}
        for i in order
        if scores[i] >= MIN_SIMILARITY
    ]


# ---------- per-question answer ----------

ANSWER_SYSTEM = f"""You answer one question about {{person}} using ONLY the source snippets the user gives you.

Rules:
- Use nothing from your own knowledge, even if you are sure it is true. If a fact is not stated in the snippets, you do not know it.
- End every sentence with the ID(s) of the snippet(s) that support it, in square brackets, e.g. "He was born in Boston [d1a2b3-c0007]."
- Snippets can be about other people. Only use what they say about {{person}}.
- If the snippets do not answer the question, reply with exactly: {NOT_FOUND}
- Keep the answer to 2-5 sentences."""


def _snippet_block(hits) -> str:
    return "\n\n".join(f"[{h['chunk_id']}]\n{h['text']}" for h in hits)


def answer(collection_id: str, person: str, question: str, kind: str, run_id=None, qnum=None) -> dict:
    # Free-form questions often say "he" or "his", so search with the person's name added.
    names = [p for p in person.lower().split() if len(p) > 2]
    search = question if any(n in question.lower() for n in names) else f"{person}: {question}"
    hits = retrieve(collection_id, search)
    retrieved = [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in hits]
    if not hits:
        prompt = f"(no LLM call: no chunk scored >= {MIN_SIMILARITY})"
        response, meta = NOT_FOUND, NO_CALL
    else:
        system = ANSWER_SYSTEM.replace("{person}", person)
        user = f"Question: {question}\n\nSource snippets:\n\n{_snippet_block(hits)}"
        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}"
        response, meta = llm.chat(system, user)
        response = response.strip()
        if NOT_FOUND.lower() in response.lower() and len(response) < 60:
            response = NOT_FOUND
    log(collection_id, kind, question, retrieved, prompt, response, meta, bool(hits), run_id, qnum)
    return {"qnum": qnum, "question": question, "retrieved": retrieved, "answer": response, "hits": hits}


# ---------- story ----------

STORY_SYSTEM = """You write a short biographical story about {person} from research notes.

Rules:
- Use ONLY facts in the notes. Add nothing from your own knowledge: no dates, places, names or events the notes don't state.
- Every sentence must end with the bracketed source ID(s) the notes give for that fact, e.g. "... in Boston [d1a2b3-c0007]." Copy IDs exactly.
- Topics the notes mark NOT IN THE SOURCES are skipped. Do not mention them.
- Write flowing prose in rough chronological order, in a few paragraphs. No headings or lists."""


def write_story(run_id: str, collection_id: str, person: str, results: list[dict]) -> str:
    found = [r for r in results if r["answer"] != NOT_FOUND]
    if not found:
        story = "The sources contain nothing that answers any of the 20 questions, so there is no story to tell."
        prompt = "(no LLM call: every question was NOT IN THE SOURCES)"
        meta = NO_CALL
    else:
        notes = "\n\n".join(f"Q{r['qnum']}. {r['question']}\nA: {r['answer']}" for r in sorted(results, key=lambda r: r["qnum"]))
        system = STORY_SYSTEM.replace("{person}", person)
        user = f"Research notes about {person}:\n\n{notes}"
        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}"
        story, meta = llm.chat(system, user)
        story = story.strip()
    log(collection_id, "story", "Weave the story from the 20 answers", [], prompt, story, meta, bool(found), run_id)
    return story


# ---------- audit ----------

AUDIT_SYSTEM = """You check a story against its sources. For each numbered sentence you get the text of the snippets it cites.
Decide whether the cited snippets state everything the sentence claims.
- "supported": every claim is stated in the cited snippets.
- "partial": some claims are in the snippets, some are not.
- "unsupported": the claims are not in the cited snippets (likely from the writer's own background knowledge).
Return JSON: {"sentences": [{"n": 1, "verdict": "supported|partial|unsupported", "outside_claims": "claims not in the snippets, or empty"}]}"""


def split_sentences(story: str) -> list[str]:
    parts = []
    for para in re.split(r"\n\s*\n", story):
        para = " ".join(para.split())
        if para:
            parts.extend(s for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z\"“‘(])", para) if s.strip())
    return parts


def audit_story(run_id: str, collection_id: str, story: str, results: list[dict]) -> dict:
    retrieved_text = {h["chunk_id"]: h["text"] for r in results for h in r["hits"]}
    sentences = split_sentences(story)
    rows = []
    for i, s in enumerate(sentences, 1):
        cited = [x.strip() for m in CITE_RE.findall(s) for x in m.split(",")]
        rows.append({
            "n": i,
            "sentence": s,
            "citations": cited,
            "invalid_citations": [x for x in cited if x not in retrieved_text],
            "verdict": None,
            "outside_claims": "",
        })

    to_judge = [r for r in rows if r["citations"] and not r["invalid_citations"]]
    if to_judge:
        used = sorted({c for r in to_judge for c in r["citations"]})
        user = "Snippets:\n\n" + "\n\n".join(f"[{c}]\n{retrieved_text[c]}" for c in used)
        user += "\n\nSentences:\n" + "\n".join(f"{r['n']}. {r['sentence']} (cites {', '.join(r['citations'])})" for r in to_judge)
        raw, meta = llm.chat(AUDIT_SYSTEM, user, json_mode=True)
        log(collection_id, "audit", "Check each story sentence against its cited chunks",
            [{"chunk_id": c_} for c_ in used], f"SYSTEM:\n{AUDIT_SYSTEM}\n\nUSER:\n{user}", raw, meta, True, run_id)
        try:
            verdicts = {v["n"]: v for v in json.loads(raw).get("sentences", [])}
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            verdicts = {}
        for r in to_judge:
            v = verdicts.get(r["n"], {})
            r["verdict"] = v.get("verdict", "unchecked")
            r["outside_claims"] = v.get("outside_claims", "")

    for r in rows:
        if not r["citations"]:
            r["flag"] = "no citation"
        elif r["invalid_citations"]:
            r["flag"] = "cites a chunk that was not retrieved"
        elif r["verdict"] in ("partial", "unsupported"):
            r["flag"] = f"{r['verdict']}: possible training-data content"
        else:
            r["flag"] = ""

    return {
        "sentences": rows,
        "total": len(rows),
        "flagged": sum(1 for r in rows if r["flag"]),
        "questions_not_in_sources": [r["qnum"] for r in results if r["answer"] == NOT_FOUND],
    }


# ---------- run orchestration ----------

def settings_snapshot(collection_id: str) -> dict:
    """What a run was made with, so it can be explained and reproduced later."""
    with engine.connect() as c:
        docs = c.execute(
            select(documents.c.id, documents.c.filename, documents.c.n_words, documents.c.n_chunks)
            .where(documents.c.collection_id == collection_id).order_by(documents.c.created_at)
        ).mappings().all()
    return {
        "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL, "top_k": TOP_K,
        "min_similarity": MIN_SIMILARITY, "chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP,
        "corpus": {
            "documents": [dict(d) for d in docs],
            "n_chunks": sum(d["n_chunks"] for d in docs),
            "n_words": sum(d["n_words"] for d in docs),
        },
    }


def start_run(collection_id: str, person: str) -> str:
    run_id = new_id("r", 8)
    with engine.begin() as c:
        c.execute(insert(runs).values(
            id=run_id, collection_id=collection_id, person=person, status="running",
            progress=0, settings=dumps(settings_snapshot(collection_id)), created_at=now(),
        ))
    threading.Thread(target=_execute, args=(run_id, collection_id, person), daemon=True).start()
    return run_id


def _set(run_id: str, **vals):
    with engine.begin() as c:
        c.execute(update(runs).where(runs.c.id == run_id).values(**vals))


def _execute(run_id: str, collection_id: str, person: str):
    try:
        done = 0
        lock = threading.Lock()

        def one(i_q):
            nonlocal done
            i, q = i_q
            res = answer(collection_id, person, q.format(person=person), "query", run_id, i)
            with lock:
                done += 1
                _set(run_id, progress=done)
            return res

        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(one, enumerate(QUESTIONS, 1)))

        story = write_story(run_id, collection_id, person, results)
        _set(run_id, story=story, progress=21)
        audit = audit_story(run_id, collection_id, story, results)
        _set(run_id, audit=dumps(audit), progress=22, status="done", finished_at=now())
    except Exception as e:  # surface the failure in the UI instead of a silent dead thread
        _set(run_id, status="error", error=f"{type(e).__name__}: {e}", finished_at=now())
