import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, LargeBinary, MetaData, String, Table, Text, create_engine,
)

from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
meta = MetaData()

collections = Table(
    "collections", meta,
    Column("id", String, primary_key=True),
    Column("person", String, nullable=False),
    Column("created_at", String),
)

documents = Table(
    "documents", meta,
    Column("id", String, primary_key=True),
    Column("collection_id", String, index=True),
    Column("filename", String),
    Column("n_words", Integer),
    Column("n_chunks", Integer),
    Column("created_at", String),
)

chunks = Table(
    "chunks", meta,
    Column("id", String, primary_key=True),
    Column("collection_id", String, index=True),
    Column("document_id", String, index=True),
    Column("ord", Integer),
    Column("text", Text),
    Column("embedding", LargeBinary),  # float32 bytes, L2-normalised
)

runs = Table(
    "runs", meta,
    Column("id", String, primary_key=True),
    Column("collection_id", String, index=True),
    Column("person", String),
    Column("status", String),  # running | done | error
    Column("progress", Integer),
    Column("story", Text),
    Column("audit", Text),  # JSON
    Column("error", Text),
    Column("created_at", String),
)

# One row per LLM step: each of the 20 queries, the story, the audit, and canary questions.
query_logs = Table(
    "query_logs", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, index=True, nullable=True),
    Column("collection_id", String, index=True),
    Column("kind", String),  # query | story | audit | canary
    Column("qnum", Integer, nullable=True),
    Column("question", Text),
    Column("retrieved", Text),  # JSON [{chunk_id, score}]
    Column("prompt", Text),
    Column("response", Text),
    Column("model", String),
    Column("created_at", String),
)


def init_db():
    meta.create_all(engine)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str, n: int = 6) -> str:
    return prefix + uuid.uuid4().hex[:n]


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
