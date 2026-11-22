import hmac
import json
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import delete, func, insert, select

from . import llm, rag
from .config import APP_PASSWORD, CHAT_MODEL, EMBED_MODEL, MIN_SIMILARITY, TOP_K
from .db import chunks, collections, documents, init_db, new_id, now, query_logs, runs, engine
from .ingest import chunk_text, extract_text

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


@app.post("/api/unlock", status_code=204, dependencies=[Depends(check_pw)])
def unlock():
    """Checks the password on its own, so the page can open the rest of the app."""
    return Response(status_code=204)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {
        "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL, "top_k": TOP_K,
        "min_similarity": MIN_SIMILARITY, "password_set": bool(APP_PASSWORD),
        "questions": rag.QUESTIONS,
    }


# ---------- collections ----------

class NewCollection(BaseModel):
    person: str


@app.get("/api/collections")
def list_collections():
    with engine.connect() as c:
        rows = c.execute(select(collections).order_by(collections.c.created_at.desc())).mappings().all()
        out = []
        for r in rows:
            n = c.execute(select(func.count()).select_from(chunks).where(chunks.c.collection_id == r["id"])).scalar()
            out.append({**r, "n_chunks": n})
    return out


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
        for t in (chunks, documents):
            c.execute(delete(t).where(t.c.collection_id == cid))
        c.execute(delete(collections).where(collections.c.id == cid))
    rag.invalidate(cid)
    return {"ok": True}


@app.post("/api/collections/{cid}/upload", dependencies=[Depends(check_pw)])
async def upload(cid: str, files: list[UploadFile] = File(...)):
    with engine.connect() as c:
        if not c.execute(select(collections.c.id).where(collections.c.id == cid)).first():
            raise HTTPException(404, "No such collection")
    added = []
    for f in files:
        data = await f.read()
        text = extract_text(f.filename or "upload.txt", data)
        pieces = chunk_text(text)
        if not pieces:
            added.append({"filename": f.filename, "n_chunks": 0, "note": "no text extracted"})
            continue
        vecs = llm.embed(pieces)
        did = new_id("d")
        with engine.begin() as c:
            c.execute(insert(documents).values(
                id=did, collection_id=cid, filename=f.filename, n_words=len(text.split()),
                n_chunks=len(pieces), created_at=now(),
            ))
            c.execute(insert(chunks), [
                {"id": f"{did}-c{i:04d}", "collection_id": cid, "document_id": did, "ord": i,
                 "text": p, "embedding": v.astype(np.float32).tobytes()}
                for i, (p, v) in enumerate(zip(pieces, vecs))
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
    answers = [
        {"qnum": l["qnum"], "question": l["question"], "answer": l["response"],
         "retrieved": json.loads(l["retrieved"] or "[]"), "prompt": l["prompt"]}
        for l in _logs(run_id) if l["kind"] == "query"
    ]
    return {**r, "audit": json.loads(r["audit"]) if r["audit"] else None,
            "answers": sorted(answers, key=lambda a: a["qnum"])}


@app.get("/api/runs/{run_id}/logs.json")
def download_logs(run_id: str):
    with engine.connect() as c:
        r = c.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
    if not r:
        raise HTTPException(404, "No such run")
    body = {
        "run": {k: v for k, v in r.items() if k != "audit"},
        "audit": json.loads(r["audit"]) if r["audit"] else None,
        "log": [{**l, "retrieved": json.loads(l["retrieved"] or "[]")} for l in _logs(run_id)],
    }
    return JSONResponse(body, headers={"Content-Disposition": f'attachment; filename="run-{run_id}-logs.json"'})


# ---------- canary / free question ----------

class Ask(BaseModel):
    question: str


@app.post("/api/collections/{cid}/ask", dependencies=[Depends(check_pw)])
def ask(cid: str, body: Ask):
    with engine.connect() as c:
        col = c.execute(select(collections).where(collections.c.id == cid)).mappings().first()
    if not col:
        raise HTTPException(404, "No such collection")
    res = rag.answer(cid, col["person"], body.question.strip(), "canary")
    return {"question": res["question"], "answer": res["answer"], "retrieved": res["retrieved"]}


@app.get("/api/collections/{cid}/canary-logs")
def canary_logs(cid: str):
    with engine.connect() as c:
        rows = c.execute(
            select(query_logs).where(query_logs.c.collection_id == cid, query_logs.c.kind == "canary").order_by(query_logs.c.id.desc())
        ).mappings().all()
    return [{**r, "retrieved": json.loads(r["retrieved"] or "[]")} for r in rows]


@app.get("/api/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    with engine.connect() as c:
        r = c.execute(
            select(chunks.c.id, chunks.c.text, chunks.c.ord, documents.c.filename)
            .join(documents, documents.c.id == chunks.c.document_id)
            .where(chunks.c.id == chunk_id)
        ).mappings().first()
    if not r:
        raise HTTPException(404, "No such chunk")
    return dict(r)
