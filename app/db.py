import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, LargeBinary, MetaData, String, Table, Text, create_engine, inspect, text,
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
    Column("settings", Text),  # JSON snapshot: models, retrieval settings, corpus at run time
    Column("created_at", String),
    Column("finished_at", String),
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
    Column("llm_called", Integer),  # 0 when nothing cleared the similarity threshold
    Column("tokens_in", Integer),
    Column("tokens_out", Integer),
    Column("latency_ms", Integer),
    Column("created_at", String),
)


def init_db():
    meta.create_all(engine)
    _add_missing_columns()


def _add_missing_columns():
    """create_all never alters an existing table, so columns added later are added here."""
    insp = inspect(engine)
    with engine.begin() as c:
        for table in meta.sorted_tables:
            have = {col["name"] for col in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    kind = col.type.compile(dialect=engine.dialect)
                    c.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {kind}'))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str, n: int = 6) -> str:
    return prefix + uuid.uuid4().hex[:n]


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
