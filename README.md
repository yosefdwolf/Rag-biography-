# Biography RAG

Live: https://rag.graydates.com (password protected; the grader has the password)

Upload any collection of pages about one person. The app chunks and embeds them, runs 20 fixed questions against the store, and writes a story using only what it retrieved. Every sentence carries the chunk IDs it came from and an exact quote that proves it. Anything the sources don't back is removed and logged. The hand-in material is in [`submission/`](submission/).

## How it works

1. **Ingest** (`app/ingest.py`): PDF, HTML or TXT becomes plain text (Project Gutenberg boilerplate is stripped). A file can hold several works, each opened by a `==== Title ====` line. Text is split into 350-word chunks with 60 words of overlap. Chunk IDs look like `d1a2b3c-c0042` (document ID plus position).
2. **Embed and store** (`app/llm.py`, `app/db.py`): each chunk is embedded with OpenAI `text-embedding-3-small` (1,536 dimensions) together with a short header, `Source: <title>` and `About: <person>`. First-person text ("I was born…") rarely names its subject, so without the header a question that names the person can't find it. Stored in Postgres (SQLite locally).
3. **Question check** (`check_question`): before searching, a model sees only the person's name and the question, never the documents. A clear question is rewritten as a standalone search query ("his" becomes the name). Code rejects a rewrite that adds any name or number not in the question. An ambiguous question ("what happened next?") gets a follow-up question instead of a guess.
4. **Retrieve** (`retrieve`): top 6 chunks by cosine similarity, kept if they score at least 0.20 and stand out from the corpus (z-score ≥ 2). If nothing stands out, no model call is made and the answer is "not in the sources".
5. **Answer** (`answer`): the model sees only the retrieved snippets. It returns claims, each with the chunk IDs and an exact quote from them, or no claims.
6. **Verify** (`verify_claims`), every claim, in the answers and the story:
   - Code: each quote must really be in a retrieved chunk (fuzzy match for ellipses and OCR noise; a wrong ID is corrected to the chunk the quote is actually in). Every number and proper name in the claim must appear in the cited chunk.
   - Model step 1 (restate): each quote is rewritten to stand on its own, with pronouns resolved from its passage, without seeing the claim. "Her name was Rowena Hamilton" becomes "Master Thomas's second wife was named Rowena Hamilton".
   - Model step 2 (judge): is the claim about the person, and do the restated statements say everything it says? Partly supported claims are trimmed to the supported part; the rest are removed. Every removal is logged with a reason.
7. **Story** (`write_story`): written only from verified claims, in the third person, tagged by stage of life and assembled in life order. Each sentence is verified again the same way.
8. **Logs**: every query, its search query, the chunk IDs and scores retrieved, the full prompts, the raw replies and every check are stored in `query_logs`. The admin page (`/admin`) shows a readable report per run, downloadable as Markdown or JSON.
9. **Canary**: any extra question goes through the same pipeline from the Canary tab and is logged.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY and APP_PASSWORD
uvicorn app.main:app --reload
```

## Deploy (Railway)

- Add a Postgres database to the project.
- Service variables: `OPENAI_API_KEY`, `APP_PASSWORD`, `DATABASE_URL=${{Postgres.DATABASE_URL}}`.
- Start command comes from `Procfile`.
- Custom domain `rag.graydates.com`: CNAME to the target Railway shows.

Secrets live only in `.env` (gitignored) and Railway variables. No key has ever been committed.
