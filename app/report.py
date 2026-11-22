"""Turns a run's raw log rows into one clean, auditable report (JSON) and a readable Markdown version."""
import json
import re
from datetime import datetime

from sqlalchemy import select

from .db import chunks, documents, engine, query_logs, runs
from .rag import CITE_RE, NOT_FOUND

PREVIEW_CHARS = 180


def _loads(s, default):
    try:
        return json.loads(s) if s else default
    except (json.JSONDecodeError, TypeError):
        return default


def _cited(text: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for m in CITE_RE.findall(text or "") for x in m.split(",")))


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
            select(chunks.c.id, chunks.c.text, chunks.c.ord, documents.c.filename)
            .join(documents, documents.c.id == chunks.c.document_id)
            .where(chunks.c.id.in_(ids))
        ).mappings().all()
    return {
        r["id"]: {"source": r["filename"], "position": r["ord"] + 1,
                  "preview": " ".join(r["text"].split())[:PREVIEW_CHARS] + "…"}
        for r in rows
    }


def _usage(row) -> dict:
    return {"tokens_in": row["tokens_in"], "tokens_out": row["tokens_out"], "latency_ms": row["latency_ms"]}


def build(run_id: str) -> dict | None:
    with engine.connect() as c:
        run = c.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
        if not run:
            return None
        rows = c.execute(select(query_logs).where(query_logs.c.run_id == run_id).order_by(query_logs.c.id)).mappings().all()
        canary_rows = c.execute(
            select(query_logs).where(query_logs.c.collection_id == run["collection_id"], query_logs.c.kind == "canary")
            .order_by(query_logs.c.id)
        ).mappings().all()

    query_rows = sorted((r for r in rows if r["kind"] == "query"), key=lambda r: r["qnum"])
    story_row = next((r for r in rows if r["kind"] == "story"), None)
    audit_row = next((r for r in rows if r["kind"] == "audit"), None)
    audit = _loads(run["audit"], None)

    all_ids = {h["chunk_id"] for r in [*query_rows, *canary_rows] for h in _loads(r["retrieved"], [])}
    info = _chunk_info(all_ids)

    def retrieved_list(row, cited):
        return [{"chunk_id": h["chunk_id"], "score": h.get("score"), "cited": h["chunk_id"] in cited,
                 **info.get(h["chunk_id"], {"source": None, "position": None, "preview": "(chunk no longer in the store)"})}
                for h in _loads(row["retrieved"], [])]

    queries = []
    for r in query_rows:
        answer = r["response"]
        cited = _cited(answer)
        got = {h["chunk_id"] for h in _loads(r["retrieved"], [])}
        called = bool(r["llm_called"]) if r["llm_called"] is not None else not r["prompt"].startswith("(no LLM call")
        queries.append({
            "qnum": r["qnum"],
            "question": r["question"],
            "outcome": "not_in_sources" if answer == NOT_FOUND else "answered",
            "llm_called": called,
            "skipped_reason": None if called else "no chunk cleared the similarity threshold",
            "retrieved": retrieved_list(r, set(cited)),
            "answer": answer,
            "citations": cited,
            "invalid_citations": [x for x in cited if x not in got],
            **_usage(r),
            "prompt": r["prompt"],
        })

    canary = []
    for r in canary_rows:
        cited = _cited(r["response"])
        canary.append({
            "asked_at": r["created_at"], "question": r["question"],
            "outcome": "not_in_sources" if r["response"] == NOT_FOUND else "answered",
            "answer": r["response"], "retrieved": retrieved_list(r, set(cited)), "prompt": r["prompt"],
        })

    sentences = (audit or {}).get("sentences", [])
    calls = [r for r in rows if r["llm_called"] or (r["llm_called"] is None and not r["prompt"].startswith("(no LLM call"))]
    tok = lambda k: sum(r[k] or 0 for r in rows)
    summary = {
        "questions": len(queries),
        "answered": sum(q["outcome"] == "answered" for q in queries),
        "not_in_sources": [q["qnum"] for q in queries if q["outcome"] == "not_in_sources"],
        "skipped_by_threshold": [q["qnum"] for q in queries if not q["llm_called"]],
        "answers_with_invalid_citations": [q["qnum"] for q in queries if q["invalid_citations"]],
        "answers_without_citations": [q["qnum"] for q in queries if q["outcome"] == "answered" and not q["citations"]],
        "story_sentences": len(sentences),
        "sentences_traced": sum(1 for s in sentences if not s.get("flag")),
        "sentences_flagged": sum(1 for s in sentences if s.get("flag")),
        "canary_questions": len(canary),
        "canary_passed": sum(c["outcome"] == "not_in_sources" for c in canary),
        "llm_calls": len(calls),
        "tokens_in": tok("tokens_in"),
        "tokens_out": tok("tokens_out"),
    }

    return {
        "report_version": 1,
        "run": {
            "id": run["id"], "person": run["person"], "status": run["status"], "error": run["error"],
            "started_at": run["created_at"], "finished_at": run["finished_at"],
            "duration_seconds": _seconds(run["created_at"], run["finished_at"]),
        },
        "settings": _loads(run["settings"], None),
        "summary": summary,
        "story": {
            "text": run["story"],
            **(_usage(story_row) if story_row else {}),
            "prompt": story_row["prompt"] if story_row else None,
        },
        "audit": {
            "method": "Each sentence must cite a chunk, every cited chunk must have been retrieved in this run, "
                      "and a separate model call checks the cited chunk text states the claim.",
            "sentences": [{
                "n": s["n"],
                "sentence": re.sub(r"\s*\[[^\]]+\]", "", s["sentence"]).strip(),
                "citations": s["citations"],
                "result": ("flagged: " + s["flag"]) if s.get("flag") else "traced",
                "verdict": s.get("verdict"),
                "claims_not_in_chunks": s.get("outside_claims") or None,
            } for s in sentences],
            "judge_prompt": audit_row["prompt"] if audit_row else None,
            "judge_response": _loads(audit_row["response"], audit_row["response"]) if audit_row else None,
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
                f"- **Retrieval:** top {st.get('top_k')} chunks per question, kept if similarity ≥ {st.get('min_similarity')}; "
                f"chunks of {st.get('chunk_words')} words with {st.get('chunk_overlap')} overlap"]
        corpus = st.get("corpus") or {}
        out.append(f"- **Corpus:** {len(corpus.get('documents', []))} document(s), {corpus.get('n_words', 0):,} words "
                   f"(~{corpus.get('n_words', 0) // 300} pages), {corpus.get('n_chunks', 0):,} chunks")
        for d in corpus.get("documents", []):
            out.append(f"  - `{d['id']}` {d['filename']} ({d['n_words']:,} words, {d['n_chunks']} chunks)")
    if run["error"]:
        out += ["", f"> **Run failed:** {run['error']}"]

    out += ["", "## Review findings", "",
            "| Check | Result |", "|---|---|",
            f"| Questions answered from the sources | {sm['answered']} / {sm['questions']} |",
            f"| Answered \"not in the sources\" | {_list(sm['not_in_sources'])} |",
            f"| Skipped the model (nothing above the threshold) | {_list(sm['skipped_by_threshold'])} |",
            f"| Answers citing a chunk that wasn't retrieved | {_list(sm['answers_with_invalid_citations'])} |",
            f"| Answers with no citation | {_list(sm['answers_without_citations'])} |",
            f"| Story sentences traced to a chunk | {sm['sentences_traced']} / {sm['story_sentences']} |",
            f"| Story sentences flagged (possible training data) | {sm['sentences_flagged']} |",
            f"| Canary questions answered \"not in the sources\" | {sm['canary_passed']} / {sm['canary_questions']} |",
            f"| Model calls · tokens in / out | {sm['llm_calls']} · {sm['tokens_in']:,} / {sm['tokens_out']:,} |"]
    flagged = [s for s in rep["audit"]["sentences"] if s["result"] != "traced"]
    if flagged:
        out += ["", "**Flagged sentences:**", ""]
        for s in flagged:
            out.append(f"- S{s['n']}: \"{s['sentence']}\" — {s['result']}"
                       + (f"; not in the cited chunk: {s['claims_not_in_chunks']}" if s["claims_not_in_chunks"] else ""))

    out += ["", "## The story", "", rep["story"]["text"] or "_(no story)_", ""]

    out += ["## Sentence trace", "", "| # | Sentence | Chunks | Result |", "|---|---|---|---|"]
    for s in rep["audit"]["sentences"]:
        out.append(f"| {s['n']} | {_cell(s['sentence'])} | {', '.join(f'`{c}`' for c in s['citations']) or '–'} | "
                   f"{_cell(s['result'])}{(' — ' + _cell(s['claims_not_in_chunks'])) if s['claims_not_in_chunks'] else ''} |")

    out += ["", "## The 20 queries", ""]
    for q in rep["queries"]:
        out += [f"### Q{q['qnum']}. {q['question']}", "",
                f"**Outcome:** {'answered' if q['outcome'] == 'answered' else 'not in the sources'}"
                + ("" if q["llm_called"] else " (model not called: no chunk cleared the threshold)")
                + (f" · {q['tokens_in']:,} in / {q['tokens_out']:,} out tokens · {q['latency_ms']} ms" if q["tokens_in"] else ""), ""]
        if q["retrieved"]:
            out += ["| Chunk | Score | Source | Cited | Preview |", "|---|---|---|---|---|"]
            for h in q["retrieved"]:
                out.append(f"| `{h['chunk_id']}` | {h['score']:.3f} | {_cell(h['source'])} | {'yes' if h['cited'] else ''} | {_cell(h['preview'])} |")
            out.append("")
        out += [f"**Answer:** {q['answer']}", ""]
        if q["invalid_citations"]:
            out += [f"> Cites chunks that were not retrieved: {', '.join(q['invalid_citations'])}", ""]
        out += ["<details><summary>Prompt sent</summary>", "", "```text", q["prompt"], "```", "", "</details>", ""]

    out += ["## Canary questions", ""]
    if not rep["canary"]:
        out.append("_None asked for this person yet._")
    for cq in rep["canary"]:
        out.append(f"- **{cq['question']}** → {'not in the sources ✓' if cq['outcome'] == 'not_in_sources' else cq['answer']}"
                   f" ({len(cq['retrieved'])} chunk(s) retrieved, asked {cq['asked_at']})")

    out += ["", "## Story prompt", "", "```text", rep["story"]["prompt"] or "", "```", ""]
    return "\n".join(out)
