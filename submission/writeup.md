# What I did

## The goal

Build a RAG app where you upload pages about one person and it writes their story from those pages only. It runs 20 fixed questions with the person's name filled in. Anything the sources don't cover comes back "not in the sources" and the story skips it. Every query, the chunk IDs retrieved and the prompt sent are logged, so each sentence can be traced back to a chunk.

## What I built

- **Live site:** https://rag.graydates.com. FastAPI on Railway with Postgres, behind a password because uploads and runs cost money.
- **Pipeline:** OpenAI `text-embedding-3-small` for embeddings and `gpt-4.1-mini` for answers. Text is split into 350-word chunks with 60 words of overlap.
- **The app:** a people page, upload, Generate story, and tabs for Story, Answers, Audit, Sources and Canary. Clicking a citation opens the chunk with the quote highlighted.
- **Admin page (`/admin`):** a readable log for each run, downloadable as Markdown or JSON.
- **Test corpora:**
  - Benjamin Franklin: his *Autobiography*, Ford's *The Many-Sided Franklin* and Fisher's *The True Benjamin Franklin*, about 319,000 words.
  - Frederick Douglass, as a second unrelated test: his *Narrative* and *My Bondage and My Freedom*, about 175,000 words.

## What went wrong and how I fixed the cause, not the symptom

The first working version cited chunks on every sentence and still let things through. Reviewing the logs and running canaries turned up four root problems.

1. **The embedding didn't know who the text was about.** Autobiographies say "I", not the person's name, so "When was Benjamin Franklin born?" missed "I was born in Boston". The person's name is only a label we give the profile. Each chunk is now embedded with a header saying which book it is from and who it is about. Before searching, each question is also rewritten as a standalone query: "his" becomes the name. That rewrite step never sees the documents, and code rejects it if it adds any name or number. A vague question like "what happened next?" gets a follow-up question instead of a guess.

2. **A citation is not proof.** The model could cite a real chunk and still write something from its own memory. For example, "His name was Benjamin Franklin [c0016]", where c0016 never says it. Now every claim must include an exact quote, and code checks that the quote really is in a retrieved chunk. Every number and name in the claim must also appear in that chunk. Then a model judge checks that the quote says what the claim says. The story is written only from claims that pass, and every story sentence is checked again.

3. **A quote cut out of its passage can lose its subject.** For Douglass, "Who was his second wife?" came back "Rowena Hamilton". The quote "Her name was Rowena Hamilton" is real, but the sentence before it is "Master Thomas married his second wife". Telling the judge to watch for this wasn't reliable, because it read the claim first and looked for confirmation. The fix is structural: one model call rewrites each quote with its subject filled in, without seeing the claim ("Master Thomas's second wife was named Rowena Hamilton"). A separate call then judges the claim against that.

4. **The checks became too strict.** Questions like "the hardest thing he went through" were rejected because the sources never use the word "hardest". Now the model picks the best-fitting passage and states its facts plainly. A label the sources don't support is trimmed off instead of throwing out the whole claim. Claims that are only general statements, like a fact about slavery in general rather than about Douglass, are removed.

## Results

On the final code, Franklin answered 18 of 20 questions and Douglass 20 of 20. All 33 and 31 story sentences passed verification. Every canary I asked whose answer is not in the books came back "not in the sources": the $100 bill, Douglass's second wife, Lincoln, John Brown and others. Details are in [review.md](review.md).

## What I would still improve

- **It leans toward removing.** 4 of Franklin's 7 removed claims were true and in the corpus. One was lost to an OCR error ("I 743" for 1743), and two came from the judge being stricter than the passage needed.
- **Order within a stage of life isn't always right.** Some answers are true but thin, like "He was a father."
- **The checks are model calls, so they can't be guaranteed.** Removals are always logged with a reason, so any slip can be seen and audited.
