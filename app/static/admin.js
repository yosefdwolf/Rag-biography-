(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const PASSWORD_KEY = 'rag-password';
  const store = {
    get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch {} },
    drop: (k) => { try { localStorage.removeItem(k); } catch {} },
  };
  const when = (iso) => (iso ? new Date(iso).toLocaleString() : '–');
  const qs = (n) => (n.length ? n.map((x) => `Q${x}`).join(', ') : 'none');

  let password = '';
  let allRuns = [];
  let current = null;

  // ---------- api & gate (same flow as the main app) ----------

  async function api(path) {
    const res = await fetch(path, { headers: { 'X-App-Password': password } });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = body.detail || {};
      if (d.reason === 'wrong_password') lockAgain('The password changed. Please enter it again.');
      throw new Error(d.message || res.statusText);
    }
    return body;
  }
  function toast(msg) {
    const t = Object.assign(document.createElement('div'), { className: 'toast bad', textContent: msg });
    $('toasts').append(t);
    setTimeout(() => t.remove(), 4500);
  }

  const gate = $('gate'), pwField = $('password'), pwError = $('passwordError');
  function showPasswordError(msg) {
    pwError.textContent = msg; pwError.hidden = false;
    pwField.setAttribute('aria-invalid', 'true'); pwField.value = ''; pwField.focus();
    const card = gate.querySelector('.gate-card');
    card.classList.remove('shake'); void card.offsetWidth; card.classList.add('shake');
  }
  pwField.addEventListener('input', () => { pwError.hidden = true; pwField.removeAttribute('aria-invalid'); });
  async function tryPassword(candidate) {
    const res = await fetch('/api/unlock', { method: 'POST', headers: { 'X-App-Password': candidate } });
    if (res.ok) return true;
    return ((await res.json().catch(() => ({}))).detail || {}).reason || 'wrong_password';
  }
  function lockAgain(message) {
    store.drop(PASSWORD_KEY);
    document.documentElement.classList.remove('known-word');
    password = ''; $('app').hidden = true; gate.hidden = false;
    showPasswordError(message);
  }
  function openApp() {
    store.set(PASSWORD_KEY, password);
    gate.hidden = true; $('app').hidden = false; $('app').classList.add('reveal');
    loadRuns().catch((e) => toast(e.message));
  }
  $('gateForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!pwField.value) return showPasswordError('Type the password first.');
    $('unlockBtn').disabled = true;
    try {
      const ok = await tryPassword(pwField.value);
      if (ok === true) { password = pwField.value; openApp(); }
      else showPasswordError(ok === 'no_app_password' ? 'This server has no password set yet, so it is locked.' : 'That password isn’t right.');
    } catch { showPasswordError('Couldn’t reach the server. Check your connection and try again.'); }
    finally { $('unlockBtn').disabled = false; }
  });

  // ---------- run list ----------

  async function loadRuns() {
    allRuns = await api('/api/admin/runs');
    const people = [...new Set(allRuns.map((r) => r.person))];
    $('personFilter').innerHTML = '<option value="">Everyone</option>' + people.map((p) => `<option>${esc(p)}</option>`).join('');
    paintRunList();
    const fromHash = location.hash.match(/^#run=(\w+)$/)?.[1];
    if (location.hash === '#canary') showCanary();
    else if (fromHash || allRuns[0]) showRun(fromHash || allRuns[0].id);
  }

  function paintRunList() {
    const who = $('personFilter').value;
    const list = allRuns.filter((r) => !who || r.person === who);
    $('runList').innerHTML = list.length ? list.map((r) => {
      const status = r.status === 'done' ? '' : `<span class="badge ${r.status === 'error' ? 'bad' : 'warn'}">${r.status}</span>`;
      const flags = r.removed == null ? '' : `<span class="badge ${r.removed ? 'bad' : 'good'}">${r.removed ? `${r.removed} removed` : 'all verified'}</span>`;
      return `<button class="side-item" type="button" data-run="${r.id}" aria-current="${current === r.id}">
        <span class="side-title"><span>${esc(r.person)}</span>${status}</span>
        <span class="side-sub">${when(r.started_at)}</span>
        <span class="side-meta"><span class="badge none">${r.answered}/${r.queries_logged} answered</span>${flags}</span>
      </button>`;
    }).join('') : '<p class="muted" style="font-size:14px">No runs yet.</p>';
    $('canaryNav').setAttribute('aria-current', String(current === 'canary'));
  }
  $('personFilter').addEventListener('change', paintRunList);
  $('runList').addEventListener('click', (e) => { const b = e.target.closest('[data-run]'); if (b) showRun(b.dataset.run); });
  $('canaryNav').addEventListener('click', showCanary);

  // ---------- one run ----------

  const chip = (id, quote) => `<button class="cite inline" data-chunk="${esc(id)}"${quote ? ` data-quote="${esc(quote)}"` : ''}>${esc(id)}</button>`;
  const withChips = (text) => esc(text).replace(/\[([A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*)\]/g,
    (_, ids) => ids.split(',').map((i) => chip(i.trim())).join(' '));
  const check = (state, label, val) => `<div class="check ${state}"><span class="icon">${state === 'ok' ? '✓' : state === 'bad' ? '!' : 'i'}</span><span>${label}</span><span class="val">${val}</span></div>`;

  async function showRun(id) {
    current = id; paintRunList();
    history.replaceState(null, '', `#run=${id}`);
    $('report').innerHTML = '<div class="empty"><strong>Loading…</strong></div>';
    let r;
    try { r = await api(`/api/runs/${id}/report`); } catch (e) { $('report').innerHTML = `<div class="empty"><strong>Couldn’t load this run</strong>${esc(e.message)}</div>`; return; }
    const s = r.summary, st = r.settings || {}, corpus = st.corpus || {};
    const removedAll = [
      ...r.queries.flatMap((q) => q.claims_removed.map((c) => ({ ...c, where: `Q${q.qnum}` }))),
      ...r.audit.sentences.filter((x) => x.status !== 'kept').map((c) => ({ ...c, where: 'Story' })),
    ];

    const head = `<div class="report-head">
      <div><p class="eyebrow">Run report</p><h1>${esc(r.run.person)}</h1>
        <div class="report-meta"><span>Run <code>${r.run.id}</code></span><span>${r.run.status}</span>
          <span>Started ${when(r.run.started_at)}</span>${r.run.duration_seconds != null ? `<span>${r.run.duration_seconds}s</span>` : ''}</div></div>
      <div class="dl"><button class="btn ghost sm" data-dl="md">Report (.md)</button><button class="btn ghost sm" data-dl="json">Logs (.json)</button></div>
    </div>${r.run.error ? `<p class="warn-line">Run failed: ${esc(r.run.error)}</p>` : ''}`;

    const canaryState = s.canary_questions ? (s.canary_passed === s.canary_questions ? 'ok' : 'bad') : 'info';
    const findings = `<section class="section"><h2>Review findings</h2><div class="checks">
      ${check(s.questions === 20 ? 'ok' : 'bad', 'All 20 queries logged with retrieved chunk IDs and prompt', `${s.questions} / 20`)}
      ${check('info', 'Answered from the sources', `${s.answered} / ${s.questions}`)}
      ${check('info', 'Answered “not in the sources”', qs(s.not_in_sources))}
      ${check('info', 'No chunk stood out, so the model wasn’t asked', qs(s.skipped_retrieval))}
      ${check('ok', 'Claims kept: quote verified in the cited chunk', s.claims_kept)}
      ${check(s.claims_removed ? 'bad' : 'ok', 'Claims removed from answers (possible training data)', s.claims_removed ? `${s.claims_removed} in ${qs(s.questions_with_removed_claims)}` : 'none')}
      ${check(s.story_sentences_removed ? 'bad' : 'ok', 'Story sentences verified / removed', `${s.story_sentences_kept} / ${s.story_sentences_removed}`)}
      ${check(canaryState, 'Canary questions answered “not in the sources”', s.canary_questions ? `${s.canary_passed} / ${s.canary_questions}` : 'none asked')}
      ${check('info', 'Tokens in / out', `${s.tokens_in.toLocaleString()} / ${s.tokens_out.toLocaleString()}`)}
    </div></section>`;

    const caught = removedAll.length ? `<section class="section"><h2>Caught and removed</h2>
      <p class="muted" style="font-size:14px;margin:-4px 0 12px">Claims the model wrote that the retrieved chunks don’t back up. These never reached the story.</p>
      <div class="table-wrap"><table class="table"><thead><tr><th>Where</th><th>Claim</th><th>Why it was removed</th><th>Cited</th></tr></thead><tbody>
      ${removedAll.map((c) => `<tr class="flag-row"><td class="mono">${c.where}</td><td class="sentence">${esc(c.text)}</td><td>${esc(c.reason)}</td>
        <td>${(c.chunk_ids || []).map((x) => chip(x, (c.quotes || [])[0])).join(' ') || '–'}</td></tr>`).join('')}
      </tbody></table></div></section>` : '';

    const settings = `<section class="section"><h2>Settings &amp; corpus</h2><dl class="kv">
      <dt>Writer model</dt><dd><code class="mono-sm">${esc(st.chat_model ?? '–')}</code></dd>
      <dt>Embedding model</dt><dd><code class="mono-sm">${esc(st.embed_model ?? '–')}</code> · 1,536 dimensions</dd>
      <dt>Retrieval</dt><dd>${st.top_k ? `Top ${st.top_k} chunks per question, kept if similarity ≥ ${st.min_similarity}${st.min_z != null ? ` and z ≥ ${st.min_z}` : ''}` : '–'}</dd>
      <dt>Chunking</dt><dd>${st.chunk_words ? `${st.chunk_words} words, ${st.chunk_overlap} overlap` : '–'}</dd>
      <dt>Corpus</dt><dd>${corpus.n_chunks != null ? `${(corpus.documents || []).length} document(s) · ${corpus.n_words.toLocaleString()} words (~${Math.round(corpus.n_words / 300)} pages) · ${corpus.n_chunks.toLocaleString()} chunks` : 'Not recorded'}</dd>
      ${(corpus.documents || []).map((d) => `<dt></dt><dd><code class="mono-sm">${d.id}</code> ${esc(d.filename)}</dd>`).join('')}
    </dl></section>`;

    const story = `<section class="section"><h2>Story</h2><div class="mini-story">${
      r.story.text ? r.story.text.split(/\n\s*\n/).map((p) => `<p>${withChips(p)}</p>`).join('') : '<p class="muted">No story.</p>'}</div></section>`;

    const trace = `<section class="section"><h2>Sentence trace</h2><div class="table-wrap"><table class="table">
      <thead><tr><th>#</th><th>Sentence</th><th>Proof (exact quote)</th><th>Result</th></tr></thead><tbody>
      ${r.audit.sentences.map((x) => `<tr class="${x.status !== 'kept' ? 'flag-row' : ''}"><td class="mono">${x.n}</td><td class="sentence">${esc(x.text)}</td>
        <td>${x.quotes.map((q) => `<span class="quote">“${esc(q)}”</span>`).join('')}${x.chunk_ids.map((c) => chip(c, x.quotes[0])).join(' ') || '<span class="muted">none</span>'}</td>
        <td>${x.status === 'kept' ? '<span class="badge good">Verified</span>' : `<span class="badge bad">${x.status === 'flagged' ? 'Flagged' : 'Removed'}</span><span class="why">${esc(x.reason)}</span>`}</td></tr>`).join('')
        || '<tr><td colspan="4" class="muted">No audit for this run.</td></tr>'}
      </tbody></table></div>
      ${r.audit.judge_prompt ? `<details class="prompt-toggle"><summary>Judge prompt &amp; response</summary><pre>${esc(r.audit.judge_prompt)}</pre><pre>${esc(r.audit.judge_raw)}</pre></details>` : ''}
    </section>`;

    const queries = `<section class="section"><h2>The 20 queries</h2><div class="table-wrap"><table class="table" id="qtable">
      <thead><tr><th>Q</th><th>Question</th><th>Outcome</th><th class="num">Top z</th><th class="num">Chunks</th><th class="num">Kept</th><th class="num">Removed</th></tr></thead><tbody>
      ${r.queries.map((q, i) => {
        const zs = q.retrieved.map((h) => h.z).filter((z) => z != null);
        const top = zs.length ? Math.max(...zs).toFixed(1) : '–';
        const outcome = q.outcome === 'answered' ? '<span class="badge good">Answered</span>'
          : `<span class="badge none">Not in sources</span>${q.llm_called ? '' : '<span class="why">nothing stood out</span>'}`;
        return `<tr class="qrow" data-i="${i}"><td class="mono"><span class="chev">›</span> ${q.qnum}</td><td>${esc(q.question)}</td><td>${outcome}</td>
          <td class="num mono">${top}</td><td class="num">${q.retrieved.length}</td><td class="num">${q.claims_kept.length}</td>
          <td class="num" ${q.claims_removed.length ? 'style="color:var(--bad);font-weight:600"' : ''}>${q.claims_removed.length}</td></tr>
          <tr class="qdetail" data-for="${i}" hidden><td colspan="7">
            ${q.search_query && q.search_query !== q.question ? `<p class="muted" style="font-size:13px;margin:0 0 8px">Searched as: <em>${esc(q.search_query)}</em>${q.question_check && q.question_check.rewrite_rejected ? ` · ${esc(q.question_check.rewrite_rejected)}` : ''}</p>` : ''}
            <p class="sub-label" style="margin-top:0">Answer</p>
            <p class="answer-text${q.outcome === 'answered' ? '' : ' nf'}">${q.outcome === 'answered' ? withChips(q.answer) : 'Not in the sources.'}</p>
            ${q.claims_kept.some((c) => c.quotes.length) ? `<p class="sub-label">Proof for each claim</p><ul class="proofs">${q.claims_kept.map((c) =>
              `<li>${esc(c.text)}<span class="quote">“${esc(c.quotes.join(' … '))}”</span>${c.chunk_ids.map((x) => chip(x, c.quotes[0])).join(' ')}</li>`).join('')}</ul>` : ''}
            ${q.claims_removed.length ? `<p class="sub-label">Removed</p><ul class="removed">${q.claims_removed.map((c) =>
              `<li><s>${esc(c.text)}</s><span class="why">${esc(c.reason)}</span></li>`).join('')}</ul>` : ''}
            <p class="sub-label">Retrieved chunks</p>
            ${q.retrieved.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Chunk</th><th class="num">Score</th><th class="num">z</th><th>Source</th><th>Cited</th><th>Preview</th></tr></thead><tbody>
              ${q.retrieved.map((h) => `<tr><td>${chip(h.chunk_id)}</td><td class="num mono">${h.score.toFixed(3)}</td><td class="num mono">${h.z ?? '–'}</td><td>${esc(h.source ?? '–')}${h.position ? ` <span class="muted">#${h.position}</span>` : ''}</td>
                <td>${h.cited ? '<span class="yes">✓</span>' : ''}</td><td class="preview">${esc(h.preview)}</td></tr>`).join('')}
            </tbody></table></div>` : `<p class="muted" style="font-size:14px">No chunk stood out for this question${q.retrieval ? ` (corpus average ${q.retrieval.corpus_mean}, needed z ≥ ${q.retrieval.min_z})` : ''}, so the model wasn’t asked.</p>`}
            <details class="prompt-toggle"><summary>Prompt sent</summary><pre>${esc(q.prompt)}</pre></details>
            ${q.raw_response ? `<details class="prompt-toggle"><summary>Model’s raw reply (before checks)</summary><pre>${esc(q.raw_response)}</pre></details>` : ''}
            ${q.judge_prompt ? `<details class="prompt-toggle"><summary>Judge prompt &amp; response</summary><pre>${esc(q.judge_prompt)}</pre><pre>${esc(q.judge_raw)}</pre></details>` : ''}
            ${q.latency_ms ? `<p class="muted" style="font-size:12.5px;margin:10px 0 0">${q.tokens_in} tokens in · ${q.tokens_out} out · ${q.latency_ms} ms across this question’s model calls</p>` : ''}
          </td></tr>`;
      }).join('')}
      </tbody></table></div></section>`;

    const canary = `<section class="section"><h2>Canary questions for ${esc(r.run.person)}</h2>${canaryTable(r.canary.map((c) => ({ ...c, person: r.run.person })), false)}</section>`;
    const storyPrompt = r.story.prompt ? `<section class="section"><details class="prompt-toggle"><summary>Story prompt</summary><pre>${esc(r.story.prompt)}</pre></details>
      ${r.story.raw_response ? `<details class="prompt-toggle"><summary>Story writer’s raw reply (before checks)</summary><pre>${esc(r.story.raw_response)}</pre></details>` : ''}</section>` : '';

    $('report').innerHTML = head + findings + caught + story + trace + queries + canary + settings + storyPrompt;
    $('report').classList.add('reveal');
  }

  function canaryTable(rows, withPerson) {
    if (!rows.length) return '<div class="empty"><strong>No canary questions yet</strong>Ask one on the Canary tab in the app.</div>';
    const badge = { not_in_sources: '<span class="badge good">Not in sources</span>', follow_up: '<span class="badge none">Asked follow-up</span>' };
    return `<div class="table-wrap"><table class="table"><thead><tr>${withPerson ? '<th>Person</th>' : ''}<th>Question</th><th>Result</th><th>Answer</th><th>Retrieved</th><th>Asked</th></tr></thead><tbody>
      ${rows.map((c) => {
        const outcome = c.outcome || (c.kind === 'clarify' ? 'follow_up' : c.answer === 'NOT IN THE SOURCES' ? 'not_in_sources' : 'answered');
        return `<tr>${withPerson ? `<td>${esc(c.person)}</td>` : ''}<td>${esc(c.question)}${c.search_query && c.search_query !== c.question ? `<span class="why">searched as: ${esc(c.search_query)}</span>` : ''}</td>
          <td>${badge[outcome] || '<span class="badge warn">Answered</span>'}</td>
          <td class="sentence">${outcome === 'not_in_sources' ? '<span class="muted">–</span>' : outcome === 'follow_up' ? esc(c.answer) : withChips(c.answer)}</td>
          <td>${c.retrieved.length ? c.retrieved.map((h) => chip(h.chunk_id)).join(' ') : `<span class="muted">${outcome === 'follow_up' ? '–' : 'nothing stood out'}</span>`}</td>
          <td class="muted" style="white-space:nowrap">${when(c.asked_at)}</td></tr>`;
      }).join('')}</tbody></table></div>`;
  }

  async function showCanary() {
    current = 'canary'; paintRunList();
    history.replaceState(null, '', '#canary');
    try {
      const rows = await api('/api/admin/canary');
      $('report').innerHTML = `<div class="report-head"><div><p class="eyebrow">All people</p><h1>Canary log</h1>
        <div class="report-meta"><span>Questions that are true about the person but not in the corpus. The right answer is “not in the sources”.</span></div></div></div>
        ${canaryTable(rows, true)}`;
    } catch (e) { toast(e.message); }
  }

  // ---------- interactions ----------

  $('report').addEventListener('click', (e) => {
    const row = e.target.closest('.qrow');
    if (row && !e.target.closest('[data-chunk]')) {
      const detail = $('report').querySelector(`.qdetail[data-for="${row.dataset.i}"]`);
      detail.hidden = !detail.hidden;
      row.classList.toggle('open', !detail.hidden);
    }
    const dl = e.target.closest('[data-dl]');
    if (dl && current && current !== 'canary') download(`/api/runs/${current}/report.${dl.dataset.dl}`).catch((err) => toast(err.message));
  });

  async function download(path) {
    const res = await fetch(path, { headers: { 'X-App-Password': password } });
    if (!res.ok) throw new Error('Download failed.');
    const name = (res.headers.get('Content-Disposition') || '').match(/filename="([^"]+)"/)?.[1] || 'download';
    const url = URL.createObjectURL(await res.blob());
    const a = Object.assign(document.createElement('a'), { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const drawer = $('drawer'), scrim = $('scrim');
  async function openChunk(id, quote) {
    $('drawerTitle').textContent = id; $('drawerMeta').textContent = 'Loading…'; $('drawerBody').textContent = '';
    drawer.classList.add('open'); drawer.setAttribute('aria-hidden', 'false'); scrim.hidden = false;
    try {
      const c = await api(`/api/chunks/${encodeURIComponent(id)}`);
      $('drawerMeta').textContent = `${c.section || c.filename} · chunk ${c.ord + 1}`;
      $('drawerBody').innerHTML = `<p>${markQuote(esc(c.text), quote)}</p>`;
      $('drawerBody').querySelector('mark')?.scrollIntoView({ block: 'center' });
    } catch (e) { $('drawerMeta').textContent = 'Not found'; $('drawerBody').textContent = e.message; }
  }
  // Highlight the quoted words in the chunk, ignoring spacing, case and line-break hyphens.
  function markQuote(html, quote) {
    if (!quote) return html;
    const words = esc(quote).split(/\s+/).filter(Boolean).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    if (words.length < 2) return html;
    return html.replace(new RegExp(words.join('[\\s\\-]+'), 'i'), (m) => `<mark>${m}</mark>`);
  }
  const closeChunk = () => { drawer.classList.remove('open'); drawer.setAttribute('aria-hidden', 'true'); scrim.hidden = true; };
  document.addEventListener('click', (e) => { const c = e.target.closest('[data-chunk]'); if (c) { e.preventDefault(); openChunk(c.dataset.chunk, c.dataset.quote); } });
  $('drawerClose').addEventListener('click', closeChunk);
  scrim.addEventListener('click', closeChunk);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeChunk(); });

  // ---------- on load ----------

  (async () => {
    const saved = store.get(PASSWORD_KEY);
    if (!saved) { pwField.focus({ preventScroll: true }); return; }
    gate.hidden = true;
    const showGate = () => { store.drop(PASSWORD_KEY); document.documentElement.classList.remove('known-word'); gate.hidden = false; pwField.focus(); };
    try { if ((await tryPassword(saved)) === true) { password = saved; openApp(); } else showGate(); } catch { showGate(); }
  })();
})();
