# Biography RAG

Live: https://rag.graydates.com

Upload pages about one person. The app chunks and embeds them, runs 20 fixed questions against the store, and writes a story using only the retrieved snippets. Every sentence cites the chunk IDs it came from, and an audit pass flags any sentence that isn't backed by its cited chunks.

## How it works

1. **Ingest** (`app/ingest.py`): PDF, HTML or TXT is reduced to plain text (Project Gutenberg boilerplate is stripped), then split into 350-word chunks with 60 words of overlap. Chunk IDs look like `d1a2b3-c0042` (document ID plus position).
2. **Embed and store** (`app/llm.py`, `app/db.py`): chunks are embedded with OpenAI `text-embedding-3-small` and stored in Postgres (SQLite locally).
3. **Retrieve** (`app/rag.py`): each question, with the person's name filled in, is embedded. The top 6 chunks by cosine similarity are kept if they score at least 0.30.
4. **Answer**: the model gets only those snippets and must cite `[chunk-id]` on every sentence, or reply `NOT IN THE SOURCES`. If nothing clears the threshold, no model call is made.
5. **Story**: the 20 cited answers are woven into one narrative. Unanswered topics are skipped.
6. **Audit**: the story is split into sentences. Each one is checked for a citation, for citations that point at chunks this run actually retrieved, and by a model judge asking whether the cited chunk text states the claim. Anything else is flagged as possible training-data content.
7. **Logs**: every query, the chunk IDs and scores it retrieved, the full prompt and the response are stored in `query_logs` and downloadable per run as JSON.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
uvicorn app.main:app --reload
```

## Deploy (Railway)

- Add a Postgres database to the project.
- Service variables: `OPENAI_API_KEY`, `DATABASE_URL=${{Postgres.DATABASE_URL}}`.
- Start command comes from `Procfile`.
- Custom domain `rag.graydates.com`: CNAME to the target Railway shows.

Secrets live only in `.env` (gitignored) and Railway variables.
