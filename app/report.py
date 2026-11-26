"""Turns a run's raw log rows into one clean, auditable report (JSON) and a readable Markdown version."""
import json
import re
from datetime import datetime

from sqlalchemy import select

from .db import chunks, documents, engine, query_logs, runs
from .rag import NOT_FOUND

PREVIEW_CHARS = 180


def _loads(s, default):
    try:
        return json.loads(s) if s else default
    except (json.JSONDecodeError, TypeError):
        return default


def _seconds(a, b):
    try:
        return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())
    except (TypeError, ValueError):
        return None


def _chunk_info(ids: set[str]) -> dict:
    if not ids:
        return {}
    with engine.connect() as c:
        rows = c.execute(
            select(chunks.c.id, chunks.c.text, chunks.c.ord, chunks.c.section, documents.c.filename)
            .join(documents, documents.c.id == chunks.c.document_id).where(chunks.c.id.in_(ids))
        ).mappings().all()
    return {r["id"]: {"source": r["section"] or r["filename"], "file": r["filename"], "position": r["ord"] + 1,
                      "preview": " ".join(r["text"].split())[:PREVIEW_CHARS] + "…"} for r in rows}


def _usage(row) -> dict:
    return {"tokens_in": row["tokens_in"], "tokens_out": row["tokens_out"], "latency_ms": row["latency_ms"]}


def _claim(c: dict) -> dict:
    """One claim or story sentence, in the same shape whether it came from a new or an older run."""
    if "status" in c:  # verified pipeline
        return {"n": c.get("n"), "text": c.get("text"), "chunk_ids": c.get("chunk_ids", []), "quotes": c.get("quotes", []),
                "status": c["status"], "reason": c.get("reason"), "verdict": c.get("verdict"), "paragraph": c.get("paragraph"),
                "note": c.get("note"), "original_text": c.get("original_text"), "standalone": c.get("standalone", [])}
    flag = c.get("flag")  # older runs: flagged, not removed
    return {"n": c.get("n"), "text": re.sub(r"\s*\[[^\]]+\]", "", c.get("sentence", "")).strip(), "chunk_ids": c.get("citations", []),
            "quotes": [], "status": "flagged" if flag else "kept",
            "reason": (flag + (f"; {c['outside_claims']}" if c.get("outside_claims") else "")) if flag else None,
            "verdict": c.get("verdict"), "paragraph": None}


def build(run_id: str) -> dict | None:
    with engine.connect() as c:
        run = c.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
        if not run:
            return None
        rows = c.execute(select(query_logs).where(query_logs.c.run_id == run_id).order_by(query_logs.c.id)).mappings().all()
        canary_rows = c.execute(
            select(query_logs).where(query_logs.c.collection_id == run["collection_id"], query_logs.c.kind.in_(["canary", "clarify"]))
            .order_by(query_logs.c.id)
        ).mappings().all()

    query_rows = sorted((r for r in rows if r["kind"] == "query"), key=lambda r: r["qnum"])
    story_row = next((r for r in rows if r["kind"] == "story"), None)
    old_audit_row = next((r for r in rows if r["kind"] == "audit"), None)
    audit = _loads(run["audit"], {}) or {}

    info = _chunk_info({h["chunk_id"] for r in [*query_rows, *canary_rows] for h in _loads(r["retrieved"], [])})

    def retrieved_list(row, cited):
        return [{"chunk_id": h["chunk_id"], "score": h.get("score"), "z": h.get("z"), "cited": h["chunk_id"] in cited,
                 **info.get(h["chunk_id"], {"source": None, "file": None, "position": None, "preview": "(chunk no longer in the store)"})}
                for h in _loads(row["retrieved"], [])]

    queries = []
    for r in query_rows:
        checks = _loads(r["checks"], {}) or {}
        claims = [_claim(c) for c in checks.get("claims", [])]
        kept = [c for c in claims if c["status"] == "kept"]
        cited = {i for c in kept for i in c["chunk_ids"]} or set(re.findall(r"[a-z0-9]+-c\d{4}", r["response"] or ""))
        called = bool(r["llm_called"]) if r["llm_called"] is not None else not (r["prompt"] or "").startswith("(no")
        queries.append({
            "qnum": r["qnum"], "question": r["question"], "search_query": r["search_query"],
            "question_check": checks.get("question_check"),
            "outcome": "not_in_sources" if r["response"] == NOT_FOUND else "answered",
            "llm_called": called,
            "retrieval": checks.get("retrieval"),
            "retrieved": retrieved_list(r, cited),
            "answer": r["response"],
            "claims_kept": kept,
            "claims_removed": [c for c in claims if c["status"] != "kept"],
            **_usage(r),
            "prompt": r["prompt"], "raw_response": r["raw_response"],
            "judge_prompt": checks.get("judge_prompt"), "judge_raw": checks.get("judge_raw"),
        })

    canary = []
    for r in canary_rows:
        checks = _loads(r["checks"], {}) or {}
        claims = [_claim(c) for c in checks.get("claims", [])]
        cited = {i for c in claims if c["status"] == "kept" for i in c["chunk_ids"]}
        canary.append({
            "asked_at": r["created_at"], "question": r["question"], "search_query": r["search_query"],
            "outcome": "follow_up" if r["kind"] == "clarify" else ("not_in_sources" if r["response"] == NOT_FOUND else "answered"),
            "answer": r["response"], "retrieved": retrieved_list(r, cited),
            "claims_removed": [c for c in claims if c["status"] != "kept"], "prompt": r["prompt"],
        })

    sentences = [_claim(s) for s in audit.get("sentences", [])]
    story_checks = _loads(story_row["checks"], {}) if story_row else {}
    all_removed = [c for q in queries for c in q["claims_removed"]]
    tok = lambda k: sum(r[k] or 0 for r in rows)
    answered_canary = [c for c in canary if c["outcome"] != "follow_up"]
    summary = {
        "questions": len(queries),
        "answered": sum(q["outcome"] == "answered" for q in queries),
        "not_in_sources": [q["qnum"] for q in queries if q["outcome"] == "not_in_sources"],
        "skipped_retrieval": [q["qnum"] for q in queries if not q["llm_called"]],
        "claims_kept": sum(len(q["claims_kept"]) for q in queries),
        "claims_removed": len(all_removed),
        "questions_with_removed_claims": [q["qnum"] for q in queries if q["claims_removed"]],
        "story_sentences": len(sentences),
        "story_sentences_kept": sum(s["status"] == "kept" for s in sentences),
        "story_sentences_removed": sum(s["status"] != "kept" for s in sentences),
        "story_notes": audit.get("notes"),
        "story_notes_unused": audit.get("notes_unused"),
        "canary_questions": len(answered_canary),
        "canary_passed": sum(c["outcome"] == "not_in_sources" for c in answered_canary),
        "llm_calls": None,
        "tokens_in": tok("tokens_in"),
        "tokens_out": tok("tokens_out"),
    }

    return {
        "report_version": 2,
        "run": {"id": run["id"], "person": run["person"], "status": run["status"], "error": run["error"],
                "started_at": run["created_at"], "finished_at": run["finished_at"],
                "duration_seconds": _seconds(run["created_at"], run["finished_at"])},
        "settings": _loads(run["settings"], None),
        "summary": summary,
        "story": {"text": run["story"], **(_usage(story_row) if story_row else {}),
                  "prompt": story_row["prompt"] if story_row else None,
                  "raw_response": story_row["raw_response"] if story_row else None},
        "audit": {
            "method": "Every claim must quote a retrieved chunk word for word; code checks the quote is really in the chunk "
                      "and that every number and name in the claim appears there; a judge model checks the quote states "
                      "the claim. Claims that fail are removed and listed here with the reason.",
            "sentences": sentences,
            "judge_prompt": story_checks.get("judge_prompt") or (old_audit_row["prompt"] if old_audit_row else None),
            "judge_raw": story_checks.get("judge_raw") or (old_audit_row["response"] if old_audit_row else None),
        },
        "queries": queries,
        "canary": canary,
    }


# ---------- Markdown ----------

def _cell(s) -> str:
    return str(s if s is not None else "").replace("|", "\\|").replace("\n", " ")


def _list(nums) -> str:
    return ", ".join(f"Q{n}" for n in nums) if nums else "none"


def to_markdown(rep: dict) -> str:
    run, st, sm = rep["run"], rep["settings"] or {}, rep["summary"]
    out = [f"# Run report: {run['person']}", ""]
    out += [f"- **Run:** `{run['id']}` · {run['status']}",
            f"- **Started:** {run['started_at']} · **Finished:** {run['finished_at'] or '–'}"
            + (f" · {run['duration_seconds']}s" if run["duration_seconds"] is not None else "")]
    if st:
        out += [f"- **Models:** writer `{st.get('chat_model')}`, embeddings `{st.get('embed_model')}`",
                f"- **Retrieval:** top {st.get('top_k')} chunks per question, kept if similarity ≥ {st.get('min_similarity')}"
                + (f" and z ≥ {st['min_z']}" if st.get("min_z") is not None else "")
                + f"; chunks of {st.get('chunk_words')} words with {st.get('chunk_overlap')} overlap"]
        corpus = st.get("corpus") or {}
        out.append(f"- **Corpus:** {len(corpus.get('documents', []))} document(s), {corpus.get('n_words', 0):,} words "
                   f"(~{corpus.get('n_words', 0) // 300} pages), {corpus.get('n_chunks', 0):,} chunks")
        for d in corpus.get("documents", []):
            out.append(f"  - `{d['id']}` {d['filename']} ({d['n_words']:,} words, {d['n_chunks']} chunks)")
    if run["error"]:
        out += ["", f"> **Run failed:** {run['error']}"]

    out += ["", "## Review findings", "", "| Check | Result |", "|---|---|",
            f"| Questions answered from the sources | {sm['answered']} / {sm['questions']} |",
            f"| Answered \"not in the sources\" | {_list(sm['not_in_sources'])} |",
            f"| No chunk stood out, so no answer call | {_list(sm['skipped_retrieval'])} |",
            f"| Claims kept (quote verified in the chunk) | {sm['claims_kept']} |",
            f"| Claims removed (possible training data) | {sm['claims_removed']} in {_list(sm['questions_with_removed_claims'])} |",
            f"| Story sentences kept / removed | {sm['story_sentences_kept']} / {sm['story_sentences_removed']} |",
            f"| Canary questions answered \"not in the sources\" | {sm['canary_passed']} / {sm['canary_questions']} |",
            f"| Tokens in / out | {sm['tokens_in']:,} / {sm['tokens_out']:,} |"]

    removed = [(f"Q{q['qnum']}", c) for q in rep["queries"] for c in q["claims_removed"]] + \
              [("story", s) for s in rep["audit"]["sentences"] if s["status"] != "kept"]
    if removed:
        out += ["", "### What was caught and removed", "",
                "These are claims the model wrote that the sources don't back up, most likely from its own training data.", ""]
        for where, c in removed:
            out.append(f"- **{where}:** \"{c['text']}\" — {c['reason']}")

    out += ["", "## The story", "", rep["story"]["text"] or "_(no story)_", ""]

    out += ["## Sentence trace", "", "| # | Sentence | Chunks | Quote | Result |", "|---|---|---|---|---|"]
    for s in rep["audit"]["sentences"]:
        out.append(f"| {s['n']} | {_cell(s['text'])} | {', '.join(f'`{c}`' for c in s['chunk_ids']) or '–'} | "
                   f"{_cell(' / '.join(s['quotes']))} | {'kept' if s['status'] == 'kept' else _cell(s['status'] + ': ' + (s['reason'] or ''))} |")

    out += ["", "## The 20 queries", ""]
    for q in rep["queries"]:
        out += [f"### Q{q['qnum']}. {q['question']}", ""]
        if q["search_query"] and q["search_query"] != q["question"]:
            out += [f"**Searched as:** {q['search_query']}", ""]
        out += [f"**Outcome:** {'answered' if q['outcome'] == 'answered' else 'not in the sources'}"
                + ("" if q["llm_called"] else " (no chunk stood out, so the model wasn't asked)")
                + (f" · {q['tokens_in']:,} in / {q['tokens_out']:,} out tokens · {q['latency_ms']} ms" if q["tokens_in"] else ""), ""]
        if q["retrieved"]:
            out += ["| Chunk | Score | z | Source | Cited | Preview |", "|---|---|---|---|---|---|"]
            for h in q["retrieved"]:
                out.append(f"| `{h['chunk_id']}` | {h['score']:.3f} | {h['z'] if h['z'] is not None else '–'} | {_cell(h['source'])} | "
                           f"{'yes' if h['cited'] else ''} | {_cell(h['preview'])} |")
            out.append("")
        out += [f"**Answer:** {q['answer']}", ""]
        for c in q["claims_kept"]:
            if c["quotes"]:
                out.append(f"- \"{c['text']}\" ← quote: \"{' / '.join(c['quotes'])}\" [{', '.join(c['chunk_ids'])}]")
                if c.get("original_text"):
                    out.append(f"  - trimmed from: ~~\"{c['original_text']}\"~~ ({c.get('note')})")
        for c in q["claims_removed"]:
            out.append(f"- ~~\"{c['text']}\"~~ removed: {c['reason']}")
        out += ["", "<details><summary>Prompt sent</summary>", "", "```text", q["prompt"] or "", "```", "", "</details>", ""]

    out += ["## Canary questions", ""]
    if not rep["canary"]:
        out.append("_None asked for this person yet._")
    for cq in rep["canary"]:
        res = {"not_in_sources": "not in the sources ✓", "follow_up": f"asked a follow-up: {cq['answer']}"}.get(cq["outcome"], cq["answer"])
        out.append(f"- **{cq['question']}** → {res} ({len(cq['retrieved'])} chunk(s) retrieved, asked {cq['asked_at']})")

    out += ["", "## Story prompt", "", "```text", rep["story"]["prompt"] or "", "```", ""]
    return "\n".join(out)
