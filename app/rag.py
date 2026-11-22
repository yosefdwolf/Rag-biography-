"""The RAG pipeline: check the question, retrieve chunks, answer with quotes, verify every claim, write the story.

Nothing reaches the story unless code has confirmed its quote is really in a retrieved chunk, and a judge
has confirmed the quote states the claim. What fails is removed and logged with the reason.
"""
import json
import re
import threading
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sqlalchemy import insert, select, update

from . import llm
from .config import CHAT_MODEL, CHUNK_OVERLAP, CHUNK_WORDS, EMBED_MODEL, MIN_SIMILARITY, MIN_Z, TOP_K
from .db import chunks, documents, dumps, engine, new_id, now, query_logs, runs

NOT_FOUND = "NOT IN THE SOURCES"
MIN_QUOTE_WORDS = 3

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


class Usage:
    """Adds up tokens and time across the several model calls one step can make."""

    def __init__(self):
        self.tokens_in = self.tokens_out = self.latency_ms = 0
        self.calls = 0

    def add(self, meta: dict):
        self.calls += 1
        self.tokens_in += meta.get("tokens_in") or 0
        self.tokens_out += meta.get("tokens_out") or 0
        self.latency_ms += meta.get("latency_ms") or 0

    def columns(self) -> dict:
        if not self.calls:
            return {"tokens_in": None, "tokens_out": None, "latency_ms": None}
        return {"tokens_in": self.tokens_in, "tokens_out": self.tokens_out, "latency_ms": self.latency_ms}


def chat_json(system: str, user: str, usage: Usage) -> tuple[dict, str]:
    raw, meta = llm.chat(system, user, json_mode=True)
    usage.add(meta)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    return (data if isinstance(data, dict) else {}), raw


def log(collection_id, kind, question, retrieved, prompt, response, usage: Usage, llm_called, run_id=None, qnum=None,
        search_query=None, raw_response=None, checks=None):
    with engine.begin() as c:
        c.execute(insert(query_logs).values(
            run_id=run_id, collection_id=collection_id, kind=kind, qnum=qnum, question=question,
            search_query=search_query, retrieved=dumps(retrieved), prompt=prompt, response=response,
            raw_response=raw_response, checks=dumps(checks) if checks is not None else None,
            model=CHAT_MODEL, llm_called=int(llm_called), created_at=now(), **usage.columns(),
        ))


# ---------- 1. question check ----------

CHECK_SYSTEM = """You prepare a question for a search over documents about one person. You see only the person's name and the question. You do not see the documents.

Decide:
- "clear": the question is about this person and asks for something specific enough to look up. Rewrite it as a standalone search query: replace "he", "she", "they", "his", "her" with the person's name, and you may add a few general topic words (e.g. "illness, health" for "was he ever sick"). Use ONLY words from the question, the person's name, and general topic words. NEVER add names, places, dates, numbers or any fact about the person, even if you know them.
- "ambiguous": the question has no clear subject or topic, or depends on missing context ("what happened next?", "why did he do that?"). Write one short follow-up question asking what they mean.

Return JSON: {"status": "clear" | "ambiguous", "search_query": "...", "follow_up": "..."}"""


def check_question(person: str, question: str, usage: Usage) -> dict:
    user = f"Person: {person}\nQuestion: {question}"
    data, raw = chat_json(CHECK_SYSTEM, user, usage)
    status = data.get("status") if data.get("status") in ("clear", "ambiguous") else "clear"
    search = (data.get("search_query") or "").strip()
    if status == "clear" and not search:
        search = f"{person}: {question}"
    # The rewrite may not smuggle in facts: any number or capitalised word that is not in the
    # question or the person's name means the model added knowledge, so fall back to the original.
    allowed = set(re.findall(r"[a-z0-9]+", (question + " " + person).lower()))
    added = [w for w in re.findall(r"\b(?:\d+|[A-Z][a-z]+)\b", search) if w.lower() not in allowed]
    rejected = None
    if status == "clear" and added:
        rejected = f"rewrite added {', '.join(added)}; used the original question with the name instead"
        search = f"{person}: {question}"
    return {
        "status": status, "search_query": search if status == "clear" else None,
        "follow_up": (data.get("follow_up") or "Could you say more about what you mean?").strip() if status == "ambiguous" else None,
        "rewrite_rejected": rejected, "prompt": f"SYSTEM:\n{CHECK_SYSTEM}\n\nUSER:\n{user}", "raw": raw,
    }


# ---------- 2. retrieval ----------

_cache: dict[str, tuple] = {}
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
            .where(chunks.c.collection_id == collection_id).order_by(chunks.c.id)
        ).all()
    ids = [r.id for r in rows]
    texts = [r.text for r in rows]
    mat = np.vstack([np.frombuffer(r.embedding, dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1), np.float32)
    with _cache_lock:
        _cache[collection_id] = (ids, texts, mat)
    return ids, texts, mat


def retrieve(collection_id: str, query: str, k: int = TOP_K) -> tuple[list[dict], dict]:
    """Top-k chunks that stand out for this question.

    A fixed score cut-off means different things in different corpora, so a chunk must also be an
    outlier for this question: its score has to be MIN_Z standard deviations above the corpus average.
    A vague question scores every chunk about the same, so nothing stands out and nothing is kept."""
    ids, texts, mat = _load(collection_id)
    if not ids:
        return [], {"mean": None, "std": None}
    scores = mat @ llm.embed([query])[0]
    mean, std = float(scores.mean()), float(scores.std()) or 1e-6
    hits = []
    for i in np.argsort(-scores)[:k]:
        s, z = float(scores[i]), (float(scores[i]) - mean) / std
        if s >= MIN_SIMILARITY and z >= MIN_Z:
            hits.append({"chunk_id": ids[i], "score": round(s, 4), "z": round(z, 2), "text": texts[i]})
    return hits, {"mean": round(mean, 4), "std": round(std, 4)}


# ---------- 3. verification (shared by answers and the story) ----------

def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)  # OCR line-break hyphens
    return " ".join(re.findall(r"[a-z0-9']+", s))


ID_IN_TEXT = re.compile(r"\s*[\(\[]\s*[a-z0-9]+-c\d{4}(?:\s*,\s*[a-z0-9]+-c\d{4})*\s*[\)\]]", re.I)
ELLIPSIS = re.compile(r"\.\.\.|…|\[\.\.\.\]")
MATCH_COVERAGE = 0.85  # share of a quote's words that must appear, in order, in one place in the chunk


def quote_found(quote: str, chunk_tokens: list[str]) -> bool:
    """Is this quote really in the chunk? Tolerates what real text throws at an honest quote: "..." for
    skipped words, OCR page headers in mid-sentence, and a sentence that runs past the chunk's end.
    Each piece between "..." must match 85% of its words, in order, within a tight span."""
    pieces = [p.split() for p in (_norm(x) for x in ELLIPSIS.split(quote)) if p]
    if sum(len(p) for p in pieces) < MIN_QUOTE_WORDS:
        return False
    for words in pieces:
        if len(words) < 2:
            continue
        sm = SequenceMatcher(None, words, chunk_tokens, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        if not blocks:
            return False
        # Use the densest run of blocks: matches must sit close together, not be scattered "the"s.
        best = 0
        for i in range(len(blocks)):
            got = 0
            for j in range(i, len(blocks)):
                span = blocks[j].b + blocks[j].size - blocks[i].b
                if span > len(words) * 1.5 + 12:
                    break
                got += blocks[j].size
                best = max(best, got)
        if best / len(words) < MATCH_COVERAGE:
            return False
    return True


SENTENCE_STARTERS = {"he", "she", "his", "her", "they", "their", "the", "a", "an", "in", "on", "at", "as", "after",
                     "when", "while", "during", "later", "then", "this", "that", "it", "one", "once", "there", "with",
                     "from", "for", "by", "to", "of", "and", "but", "i", "my", "mr", "mrs", "dr"}

JUDGE_SYSTEM = """You check claims against quotes. Each claim comes with exact quotes taken from source documents.
For each claim decide whether the quotes, read together, state everything the claim says.
- "supported": every part of the claim is stated in the quotes.
- "partial": part of the claim is in the quotes, part is not.
- "unsupported": the quotes don't state the claim.
Paraphrase is fine. Added details (dates, names, places, numbers, causes, feelings) that are not in the quotes are not.
Return JSON: {"results": [{"n": 1, "verdict": "supported|partial|unsupported", "missing": "what the quotes don't support, or empty"}]}"""


def verify_claims(person: str, claims: list[dict], chunk_text: dict[str, str], usage: Usage) -> tuple[list[dict], dict | None]:
    """Each claim is {"text", "chunk_ids", "quotes"}. Returns them with status kept/removed and a reason.

    Code checks first (cheap and certain): the cited chunks were retrieved, every quote is really in
    a cited chunk, and every number or proper name in the claim appears in a cited chunk.
    Then one judge call checks the quotes actually say what the claim says."""
    person_words = set(_norm(person).split())
    tokens = {i: _norm(t).split() for i, t in chunk_text.items()}
    out = []
    for n, c in enumerate(claims, 1):
        text = ID_IN_TEXT.sub("", " ".join(str(c.get("text") or "").split())).strip()
        ids = [str(i).strip() for i in (c.get("chunk_ids") or []) if str(i).strip()]
        quotes = [str(q).strip() for q in (c.get("quotes") or []) if str(q).strip()]
        row = {"n": n, "text": text, "chunk_ids": ids, "quotes": quotes, "status": "kept", "reason": None, "verdict": None}
        cited = " ".join(_norm(chunk_text[i]) for i in ids if i in chunk_text)
        bad_ids = [i for i in ids if i not in chunk_text]
        if not text:
            row["reason"] = "empty claim"
        elif not ids:
            row["reason"] = "no chunk cited"
        elif bad_ids:
            row["reason"] = f"cites chunk(s) that weren't retrieved: {', '.join(bad_ids)}"
        elif not quotes:
            row["reason"] = "no supporting quote"
        else:
            # Each quote must be in a cited chunk. If it is in another retrieved chunk instead, the model
            # only mixed up the ID: credit the right chunk and note it.
            missing = []
            for q in quotes:
                if any(quote_found(q, tokens[i]) for i in ids):
                    continue
                elsewhere = next((i for i in tokens if i not in ids and quote_found(q, tokens[i])), None)
                if elsewhere:
                    ids.append(elsewhere)
                    row["note"] = f"quote was in {elsewhere}, not the chunk cited; citation corrected"
                else:
                    missing.append(q)
            row["chunk_ids"] = ids
            cited = " ".join(_norm(chunk_text[i]) for i in ids)
            if missing:
                row["reason"] = f"quote not found in the cited chunk: \"{missing[0][:80]}\""
            else:
                cited_words = set(cited.split())
                numbers = list(dict.fromkeys(x for x in re.findall(r"\d+", " ".join([text, *quotes])) if x not in cited_words))
                words = text.split()
                names = [re.sub(r"['’]s$", "", w.strip(".,;:!?\"'()")) for k, w in enumerate(words)
                         if k > 0 and re.match(r"^[A-Z][a-z]{2,}", w.strip("\"'(")) and not words[k - 1].endswith((".", "!", "?"))]
                names = [w for w in names if w.lower() not in SENTENCE_STARTERS and w.lower() not in person_words
                         and _norm(w) and _norm(w) not in cited]
                if numbers:
                    row["reason"] = f"number not in the cited chunk: {', '.join(numbers)}"
                elif names:
                    row["reason"] = f"name not in the cited chunk: {', '.join(dict.fromkeys(names))}"
        if row["reason"]:
            row["status"] = "removed"
        out.append(row)

    judge = None
    to_judge = [r for r in out if r["status"] == "kept"]
    if to_judge:
        user = "\n\n".join(f"{r['n']}. Claim: {r['text']}\n" + "\n".join(f"   Quote: \"{q}\"" for q in r["quotes"]) for r in to_judge)
        data, raw = chat_json(JUDGE_SYSTEM, user, usage)
        verdicts = {v.get("n"): v for v in data.get("results", []) if isinstance(v, dict)}
        for r in to_judge:
            v = verdicts.get(r["n"], {})
            r["verdict"] = v.get("verdict") or "unchecked"
            if r["verdict"] in ("partial", "unsupported"):
                r["status"], r["reason"] = "removed", f"{r['verdict']}: {v.get('missing') or 'quotes do not state the claim'}"
        judge = {"prompt": f"SYSTEM:\n{JUDGE_SYSTEM}\n\nUSER:\n{user}", "raw": raw}
    return out, judge


# ---------- 4. answer one question ----------

ANSWER_SYSTEM = """You answer one question using ONLY the source snippets you are given.

The question names a person only so you know who it is about. The name is not evidence: you know nothing about this person except what the snippets say, even if you have heard of them.

Rules:
- Every claim needs at least one exact quote copied word for word from a snippet (at least 3 words), and the ID of that snippet.
- Never state a date, name, place or number unless it appears in your quote.
- Snippets can be about other people. Only use what they say about the person asked about.
- If the snippets don't answer the question, return no claims.

Return JSON: {"claims": [{"text": "one sentence answering part of the question", "chunk_ids": ["id"], "quotes": ["exact words from that snippet"]}]}"""


def _snippets(hits) -> str:
    return "\n\n".join(f"[{h['chunk_id']}]\n{h['text']}" for h in hits)


def _cited_text(claims: list[dict]) -> str:
    return " ".join(f"{c['text']} [{', '.join(c['chunk_ids'])}]" for c in claims if c["status"] == "kept")


def answer(collection_id: str, person: str, question: str, kind: str, run_id=None, qnum=None, allow_follow_up=False) -> dict:
    usage = Usage()
    check = check_question(person, question, usage)
    checks = {"question_check": {k: check[k] for k in ("status", "search_query", "follow_up", "rewrite_rejected")},
              "question_check_prompt": check["prompt"], "question_check_raw": check["raw"]}

    if check["status"] == "ambiguous" and allow_follow_up:
        log(collection_id, "clarify", question, [], check["prompt"], check["follow_up"], usage, True,
            run_id, qnum, checks=checks)
        return {"qnum": qnum, "question": question, "follow_up": check["follow_up"], "answer": None, "hits": [], "claims": []}

    search = check["search_query"] or f"{person}: {question}"
    hits, dist = retrieve(collection_id, search)
    checks["retrieval"] = {"corpus_mean": dist["mean"], "corpus_std": dist["std"], "min_similarity": MIN_SIMILARITY, "min_z": MIN_Z}
    retrieved = [{"chunk_id": h["chunk_id"], "score": h["score"], "z": h["z"]} for h in hits]

    claims, raw, prompt = [], None, None
    if not hits:
        prompt = f"(no answer call: no chunk stood out for this question: score ≥ {MIN_SIMILARITY} and z ≥ {MIN_Z})"
    else:
        user = f"Question: {question}\n\nSource snippets:\n\n{_snippets(hits)}"
        prompt = f"SYSTEM:\n{ANSWER_SYSTEM}\n\nUSER:\n{user}"
        data, raw = chat_json(ANSWER_SYSTEM, user, usage)
        proposed = [c for c in data.get("claims", []) if isinstance(c, dict)]
        claims, judge = verify_claims(person, proposed, {h["chunk_id"]: h["text"] for h in hits}, usage)
        if judge:
            checks["judge_prompt"], checks["judge_raw"] = judge["prompt"], judge["raw"]
    checks["claims"] = claims

    response = _cited_text(claims) or NOT_FOUND
    log(collection_id, kind, question, retrieved, prompt, response, usage, bool(hits), run_id, qnum,
        search_query=search, raw_response=raw, checks=checks)
    return {"qnum": qnum, "question": question, "retrieved": retrieved, "answer": response, "hits": hits,
            "claims": [c for c in claims if c["status"] == "kept"], "follow_up": None}


# ---------- 5. story ----------

STORY_SYSTEM = """You write a short biographical story about {person} from verified notes. Each note is a fact with the exact quote and source ID that prove it.

Rules:
- Use ONLY the facts in the notes. Add nothing: no dates, names, places, numbers, causes or feelings the notes don't state.
- Every sentence must list the source IDs and copy the exact quotes (from the notes) that prove it.
- Write flowing prose in rough chronological order, in a few paragraphs. Join related facts, but never invent links between them.

Return JSON: {"paragraphs": [{"sentences": [{"text": "...", "chunk_ids": ["id"], "quotes": ["exact quote from the notes"]}]}]}"""


def write_story(run_id: str, collection_id: str, person: str, results: list[dict]) -> tuple[str, dict]:
    usage = Usage()
    notes = [(r["qnum"], c) for r in sorted(results, key=lambda r: r["qnum"]) for c in r["claims"]]
    chunk_text = {h["chunk_id"]: h["text"] for r in results for h in r["hits"]}
    audit = {"sentences": [], "total": 0, "kept": 0, "removed": 0,
             "questions_not_in_sources": [r["qnum"] for r in results if r["answer"] == NOT_FOUND]}
    if not notes:
        story = "The sources contain nothing that answers any of the 20 questions, so there is no story to tell."
        log(collection_id, "story", "Write the story from the verified answers", [], "(no story call: no verified facts)",
            story, usage, False, run_id)
        return story, audit

    lines = [f"Q{q}: {c['text']}\n   source: {', '.join(c['chunk_ids'])}\n" + "\n".join(f"   quote: \"{x}\"" for x in c["quotes"])
             for q, c in notes]
    system = STORY_SYSTEM.replace("{person}", person)
    user = "Verified notes:\n\n" + "\n\n".join(lines)
    data, raw = chat_json(system, user, usage)

    flat = []
    for p_i, para in enumerate(data.get("paragraphs", []) if isinstance(data.get("paragraphs"), list) else []):
        for s in (para.get("sentences", []) if isinstance(para, dict) else []):
            if isinstance(s, dict):
                flat.append((p_i, s))
    checked, judge = verify_claims(person, [s for _, s in flat], chunk_text, usage)
    paragraphs: dict[int, list[str]] = {}
    for (p_i, _), c in zip(flat, checked):
        c["paragraph"] = p_i
        if c["status"] == "kept":
            paragraphs.setdefault(p_i, []).append(f"{c['text']} [{', '.join(c['chunk_ids'])}]")
    story = "\n\n".join(" ".join(v) for _, v in sorted(paragraphs.items())) or \
        "Every sentence the writer proposed failed verification, so there is no story to show. See the audit."

    audit.update(sentences=checked, total=len(checked), kept=sum(c["status"] == "kept" for c in checked),
                 removed=sum(c["status"] == "removed" for c in checked))
    checks = {"claims": checked, **({"judge_prompt": judge["prompt"], "judge_raw": judge["raw"]} if judge else {})}
    log(collection_id, "story", "Write the story from the verified answers", [], f"SYSTEM:\n{system}\n\nUSER:\n{user}",
        story, usage, True, run_id, raw_response=raw, checks=checks)
    return story, audit


# ---------- run orchestration ----------

def settings_snapshot(collection_id: str) -> dict:
    """What a run was made with, so it can be explained and reproduced later."""
    with engine.connect() as c:
        docs = c.execute(
            select(documents.c.id, documents.c.filename, documents.c.n_words, documents.c.n_chunks)
            .where(documents.c.collection_id == collection_id).order_by(documents.c.created_at)
        ).mappings().all()
    return {
        "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL, "top_k": TOP_K, "min_similarity": MIN_SIMILARITY,
        "min_z": MIN_Z, "min_quote_words": MIN_QUOTE_WORDS, "chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP,
        "corpus": {"documents": [dict(d) for d in docs], "n_chunks": sum(d["n_chunks"] for d in docs),
                   "n_words": sum(d["n_words"] for d in docs)},
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

        _set(run_id, progress=20)
        story, audit = write_story(run_id, collection_id, person, results)
        _set(run_id, story=story, audit=dumps(audit), progress=22, status="done", finished_at=now())
    except Exception as e:  # surface the failure in the UI instead of a silent dead thread
        _set(run_id, status="error", error=f"{type(e).__name__}: {e}", finished_at=now())
