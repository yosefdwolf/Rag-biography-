import hmac
import json
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import delete, func, insert, select

from . import llm, rag, report
from .config import APP_PASSWORD, CHAT_MODEL, EMBED_MODEL, MIN_SIMILARITY, MIN_Z, TOP_K
from .db import chunks, collections, documents, init_db, new_id, now, query_logs, runs, engine
from .ingest import chunk_text, embedding_text, extract_text, split_sections, title_from_filename

app = FastAPI(title="Biography RAG")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def _startup():
    init_db()


def password_problem(given: str | None) -> tuple[int, str, str] | None:
    """Anything that spends OpenAI credits needs APP_PASSWORD. Unset means refuse (fail closed)."""
    if not APP_PASSWORD:
        return 503, "no_app_password", "The server has no APP_PASSWORD set, so it won't spend credits."
    if not given or not hmac.compare_digest(given.encode(), APP_PASSWORD.encode()):
        return 401, "wrong_password", "Wrong password."
    return None


def check_pw(x_app_password: str | None = Header(default=None)):
    if problem := password_problem(x_app_password):
        status, reason, message = problem
        raise HTTPException(status, {"reason": reason, "message": message})


OPEN_API = {"/api/config", "/api/unlock"}


@app.middleware("http")
async def require_password(request: Request, call_next):
    """Every /api route except config and unlock needs the password, reads included (logs hold prompts)."""
    path = request.url.path
    if path.startswith("/api/") and path not in OPEN_API:
        if problem := password_problem(request.headers.get("x-app-password")):
            status, reason, message = problem
            return JSONResponse({"detail": {"reason": reason, "message": message}}, status_code=status)
    return await call_next(request)


@app.post("/api/unlock", status_code=204, dependencies=[Depends(check_pw)])
def unlock():
    """Checks the password on its own, so the page can open the rest of the app."""
    return Response(status_code=204)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC / "admin.html")


@app.get("/api/config")
def config():
    return {
        "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL, "top_k": TOP_K,
        "min_similarity": MIN_SIMILARITY, "min_z": MIN_Z, "password_set": bool(APP_PASSWORD),
        "questions": rag.QUESTIONS,
    }


# ---------- collections ----------

class NewCollection(BaseModel):
    person: str


@app.get("/api/collections")
def list_collections():
    with engine.connect() as c:
        rows = c.execute(select(collections).order_by(collections.c.created_at.desc())).mappings().all()
        docs = c.execute(select(documents.c.collection_id, func.count(), func.sum(documents.c.n_words), func.sum(documents.c.n_chunks))
                         .group_by(documents.c.collection_id)).all()
        rs = c.execute(select(runs.c.collection_id, func.count(), func.max(runs.c.created_at)).group_by(runs.c.collection_id)).all()
    d = {r[0]: r[1:] for r in docs}
    ru = {r[0]: r[1:] for r in rs}
    return [{
        **r, "n_docs": d.get(r["id"], (0,))[0], "n_words": (d.get(r["id"], (0, 0))[1] or 0),
        "n_chunks": (d.get(r["id"], (0, 0, 0))[2] or 0), "n_runs": ru.get(r["id"], (0,))[0],
        "last_run_at": ru.get(r["id"], (0, None))[1],
    } for r in rows]


@app.post("/api/collections", dependencies=[Depends(check_pw)])
def create_collection(body: NewCollection):
    person = body.person.strip()
    if not person:
        raise HTTPException(400, "Person name is required")
    cid = new_id("c")
    with engine.begin() as c:
        c.execute(insert(collections).values(id=cid, person=person, created_at=now()))
    return {"id": cid, "person": person}


@app.get("/api/collections/{cid}")
def get_collection(cid: str):
    with engine.connect() as c:
        col = c.execute(select(collections).where(collections.c.id == cid)).mappings().first()
        if not col:
            raise HTTPException(404, "No such collection")
        docs = c.execute(select(documents).where(documents.c.collection_id == cid).order_by(documents.c.created_at)).mappings().all()
        rs = c.execute(
            select(runs.c.id, runs.c.status, runs.c.created_at).where(runs.c.collection_id == cid).order_by(runs.c.created_at.desc())
        ).mappings().all()
    words = sum(d["n_words"] for d in docs)
    return {
        **col, "documents": list(docs), "runs": list(rs),
        "n_words": words, "n_chunks": sum(d["n_chunks"] for d in docs),
        "est_pages": round(words / 300),  # ~300 words per printed book page
    }


@app.delete("/api/collections/{cid}", dependencies=[Depends(check_pw)])
def delete_collection(cid: str):
    with engine.begin() as c:
        for t in (chunks, documents, runs, query_logs):
            c.execute(delete(t).where(t.c.collection_id == cid))
        c.execute(delete(collections).where(collections.c.id == cid))
    rag.invalidate(cid)
    return {"ok": True}


@app.post("/api/collections/{cid}/upload", dependencies=[Depends(check_pw)])
async def upload(cid: str, files: list[UploadFile] = File(...)):
    with engine.connect() as c:
        col = c.execute(select(collections).where(collections.c.id == cid)).mappings().first()
    if not col:
        raise HTTPException(404, "No such collection")
    added = []
    for f in files:
        data = await f.read()
        text = extract_text(f.filename or "upload.txt", data)
        # Chunk each work in the file separately so a chunk never spans two books and knows its source.
        pieces = [(title, p) for title, body in split_sections(text, title_from_filename(f.filename))
                  for p in chunk_text(body)]
        if not pieces:
            added.append({"filename": f.filename, "n_chunks": 0, "note": "no text extracted"})
            continue
        vecs = llm.embed([embedding_text(col["person"], title, p) for title, p in pieces])
        did = new_id("d")
        with engine.begin() as c:
            c.execute(insert(documents).values(
                id=did, collection_id=cid, filename=f.filename, n_words=len(text.split()),
                n_chunks=len(pieces), created_at=now(),
            ))
            c.execute(insert(chunks), [
                {"id": f"{did}-c{i:04d}", "collection_id": cid, "document_id": did, "ord": i,
                 "section": title, "text": p, "embedding": v.astype(np.float32).tobytes()}
                for i, ((title, p), v) in enumerate(zip(pieces, vecs))
            ])
        added.append({"filename": f.filename, "document_id": did, "n_chunks": len(pieces), "n_words": len(text.split())})
    rag.invalidate(cid)
    return {"added": added}


# ---------- runs ----------

@app.post("/api/collections/{cid}/runs", dependencies=[Depends(check_pw)])
def start(cid: str):
    with engine.connect() as c:
        col = c.execute(select(collections).where(collections.c.id == cid)).mappings().first()
    if not col:
        raise HTTPException(404, "No such collection")
    return {"run_id": rag.start_run(cid, col["person"])}


def _logs(run_id: str):
    with engine.connect() as c:
        return c.execute(select(query_logs).where(query_logs.c.run_id == run_id).order_by(query_logs.c.id)).mappings().all()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    with engine.connect() as c:
        r = c.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
    if not r:
        raise HTTPException(404, "No such run")
    answers = []
    for l in _logs(run_id):
        if l["kind"] != "query":
            continue
        checks = json.loads(l["checks"]) if l["checks"] else {}
        answers.append({
            "qnum": l["qnum"], "question": l["question"], "answer": l["response"], "search_query": l["search_query"],
            "retrieved": json.loads(l["retrieved"] or "[]"), "prompt": l["prompt"],
            "removed": [c for c in checks.get("claims", []) if c.get("status") == "removed"],
        })
    return {**r, "audit": json.loads(r["audit"]) if r["audit"] else None,
            "answers": sorted(answers, key=lambda a: a["qnum"])}


def _report(run_id: str) -> dict:
    rep = report.build(run_id)
    if rep is None:
        raise HTTPException(404, "No such run")
    return rep


def _filename(rep: dict, ext: str) -> str:
    slug = "-".join(rep["run"]["person"].lower().split()) or "person"
    return f"{slug}-run-{rep['run']['id']}.{ext}"


@app.get("/api/runs/{run_id}/report")
def run_report(run_id: str):
    return _report(run_id)


@app.get("/api/runs/{run_id}/report.json")
def run_report_json(run_id: str):
    rep = _report(run_id)
    return Response(json.dumps(rep, indent=2, ensure_ascii=False), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{_filename(rep, "json")}"'})


@app.get("/api/runs/{run_id}/report.md")
def run_report_md(run_id: str):
    rep = _report(run_id)
    return PlainTextResponse(report.to_markdown(rep), media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{_filename(rep, "md")}"'})


# ---------- admin ----------

@app.get("/api/admin/runs")
def admin_runs():
    with engine.connect() as c:
        rs = c.execute(select(runs).order_by(runs.c.created_at.desc())).mappings().all()
        counts = {}
        for row in c.execute(select(query_logs.c.run_id, query_logs.c.response, query_logs.c.checks)
                             .where(query_logs.c.kind == "query")).all():
            n = counts.setdefault(row.run_id, [0, 0, 0])
            n[0] += 1
            n[1] += row.response != rag.NOT_FOUND
            claims = (json.loads(row.checks) if row.checks else {}).get("claims", [])
            n[2] += sum(cl.get("status") == "removed" for cl in claims)
    out = []
    for r in rs:
        audit = json.loads(r["audit"]) if r["audit"] else None
        logged, answered, claims_removed = counts.get(r["id"], [0, 0, 0])
        out.append({
            "id": r["id"], "person": r["person"], "collection_id": r["collection_id"], "status": r["status"],
            "started_at": r["created_at"], "finished_at": r["finished_at"], "queries_logged": logged,
            "answered": answered, "sentences": audit.get("total") if audit else None,
            "removed": (audit.get("removed", audit.get("flagged")) or 0) + claims_removed if audit else None,
        })
    return out


@app.get("/api/admin/canary")
def admin_canary():
    with engine.connect() as c:
        rows = c.execute(
            select(query_logs, collections.c.person)
            .join(collections, collections.c.id == query_logs.c.collection_id)
            .where(query_logs.c.kind.in_(["canary", "clarify"])).order_by(query_logs.c.id.desc())
        ).mappings().all()
    return [{"id": r["id"], "person": r["person"], "kind": r["kind"], "question": r["question"], "answer": r["response"],
             "search_query": r["search_query"], "retrieved": json.loads(r["retrieved"] or "[]"),
             "llm_called": r["llm_called"], "asked_at": r["created_at"], "prompt": r["prompt"]} for r in rows]


# ---------- canary / free question ----------

class Ask(BaseModel):
    question: str
    clarification: str | None = None  # the user's reply to a follow-up question


@app.post("/api/collections/{cid}/ask", dependencies=[Depends(check_pw)])
def ask(cid: str, body: Ask):
    with engine.connect() as c:
        col = c.execute(select(collections).where(collections.c.id == cid)).mappings().first()
    if not col:
        raise HTTPException(404, "No such collection")
    question = body.question.strip()
    if body.clarification and body.clarification.strip():
        question = f"{question} ({body.clarification.strip()})"
    # One follow-up at most: once the person has clarified, answer whatever they meant.
    res = rag.answer(cid, col["person"], question, "canary", allow_follow_up=not body.clarification)
    if res["follow_up"]:
        return {"question": question, "follow_up": res["follow_up"]}
    return {"question": res["question"], "answer": res["answer"], "retrieved": res["retrieved"]}


@app.get("/api/collections/{cid}/canary-logs")
def canary_logs(cid: str):
    with engine.connect() as c:
        rows = c.execute(
            select(query_logs).where(query_logs.c.collection_id == cid, query_logs.c.kind == "canary").order_by(query_logs.c.id.desc())
        ).mappings().all()
    out = []
    for r in rows:
        checks = json.loads(r["checks"]) if r["checks"] else {}
        out.append({"id": r["id"], "question": r["question"], "response": r["response"], "created_at": r["created_at"],
                    "search_query": r["search_query"], "retrieved": json.loads(r["retrieved"] or "[]"),
                    "removed": [c for c in checks.get("claims", []) if c.get("status") == "removed"]})
    return out


@app.get("/api/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    with engine.connect() as c:
        r = c.execute(
            select(chunks.c.id, chunks.c.text, chunks.c.ord, chunks.c.section, documents.c.filename)
            .join(documents, documents.c.id == chunks.c.document_id)
            .where(chunks.c.id == chunk_id)
        ).mappings().first()
    if not r:
        raise HTTPException(404, "No such chunk")
    return dict(r)
