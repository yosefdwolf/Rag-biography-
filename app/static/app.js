(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const NOT_FOUND = 'NOT IN THE SOURCES';
  const PASSWORD_KEY = 'rag-password';
  const store = {
    get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch {} },
    drop: (k) => { try { localStorage.removeItem(k); } catch {} },
  };

  let password = '';
  let cfg = {};
  let col = null;       // selected collection
  let run = null;       // run on screen
  let pollTimer = 0;
  let queue = [];       // files waiting to upload

  // ---------- api ----------

  class ApiError extends Error {
    constructor(status, reason, message) { super(message); this.status = status; this.reason = reason; }
  }

  async function api(path, opts = {}) {
    opts.headers = { 'X-App-Password': password, ...(opts.headers || {}) };
    const res = await fetch(path, opts);
    if (res.status === 204) return null;
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = body.detail;
      const reason = (d && d.reason) || '';
      const msg = (d && d.message) || (typeof d === 'string' ? d : res.statusText);
      if (reason === 'wrong_password') lockAgain('The password changed. Please enter it again.');
      throw new ApiError(res.status, reason, msg);
    }
    return body;
  }
  const json = (body) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

  function toast(msg, bad = false) {
    const t = document.createElement('div');
    t.className = 'toast' + (bad ? ' bad' : '');
    t.textContent = msg;
    $('toasts').append(t);
    setTimeout(() => t.remove(), 4500);
  }
  const fail = (e) => { if (e.reason !== 'wrong_password') toast(e.message || 'Something went wrong.', true); };

  function busy(btn, on, label) {
    if (on) { btn.dataset.label = btn.innerHTML; btn.innerHTML = `<span class="spinner"></span>${esc(label || 'Working…')}`; btn.disabled = true; }
    else { btn.innerHTML = btn.dataset.label || btn.innerHTML; btn.disabled = false; }
  }

  // ---------- password gate ----------

  const gate = $('gate'), pwField = $('password'), pwError = $('passwordError');

  function showPasswordError(msg) {
    pwError.textContent = msg;
    pwError.hidden = false;
    pwField.setAttribute('aria-invalid', 'true');
    pwField.value = '';
    pwField.focus();
    const card = gate.querySelector('.gate-card');
    card.classList.remove('shake'); void card.offsetWidth; card.classList.add('shake');
  }
  pwField.addEventListener('input', () => { pwError.hidden = true; pwField.removeAttribute('aria-invalid'); });

  async function tryPassword(candidate) {
    const res = await fetch('/api/unlock', { method: 'POST', headers: { 'X-App-Password': candidate } });
    if (res.ok) return true;
    const d = (await res.json().catch(() => ({}))).detail || {};
    return d.reason || 'wrong_password';
  }

  function openApp() {
    store.set(PASSWORD_KEY, password);
    gate.hidden = true;
    $('app').hidden = false;
    $('app').classList.add('reveal');
    $('topActions').hidden = false;
    loadCols();
  }

  function lockAgain(message) {
    store.drop(PASSWORD_KEY);
    document.documentElement.classList.remove('known-word');
    password = '';
    clearTimeout(pollTimer);
    $('app').hidden = true;
    $('topActions').hidden = true;
    gate.hidden = false;
    showPasswordError(message);
  }

  $('gateForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!pwField.value) return showPasswordError('Type the password first.');
    const btn = $('unlockBtn');
    busy(btn, true, 'Checking…');
    try {
      const ok = await tryPassword(pwField.value);
      if (ok === true) { password = pwField.value; openApp(); }
      else if (ok === 'no_app_password') showPasswordError('This server has no password set yet, so it is locked.');
      else showPasswordError('That password isn’t right.');
    } catch {
      showPasswordError('Couldn’t reach the server. Check your connection and try again.');
    } finally { busy(btn, false); }
  });

  // ---------- people ----------

  // The address says what's on screen: #/ is the people list, #/p/<id> is one person.
  let people = [];
  const routeId = () => (location.hash.match(/^#\/p\/(\w+)$/) || [])[1];

  async function loadCols() {
    people = await api('/api/collections');
    paintPicker();
    await route();
  }

  function paintPicker() {
    const sel = $('colSel');
    sel.innerHTML = '<option value="">All people</option>' + people.map((c) => `<option value="${c.id}">${esc(c.person)}</option>`).join('');
    sel.hidden = !people.length;
  }

  async function route() {
    const id = routeId();
    clearTimeout(pollTimer);
    if (id && people.some((c) => c.id === id)) return selectCol(id);
    showHome();
  }
  window.addEventListener('hashchange', () => { if (password) route().catch(fail); });

  function showHome() {
    col = null; run = null;
    $('colSel').value = '';
    $('person').hidden = true;
    $('welcome').hidden = people.length > 0;
    $('people').hidden = people.length === 0;
    const ago = (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : null);
    $('peopleGrid').innerHTML = people.map((c) => `
      <a class="person-card" href="#/p/${c.id}">
        <span class="avatar" aria-hidden="true">${esc(c.person.split(/\s+/).map((w) => w[0]).slice(0, 2).join('').toUpperCase())}</span>
        <span class="pc-name">${esc(c.person)}</span>
        <span class="pc-stats">${c.n_docs} doc${c.n_docs === 1 ? '' : 's'} · ~${Math.round(c.n_words / 300).toLocaleString()} pages · ${c.n_chunks.toLocaleString()} chunks</span>
        <span class="pc-foot">${c.n_runs ? `${c.n_runs} run${c.n_runs === 1 ? '' : 's'} · last ${ago(c.last_run_at)}` : (c.n_chunks ? 'Ready to generate' : 'No sources yet')}</span>
      </a>`).join('');
  }

  async function selectCol(id) {
    clearTimeout(pollTimer);
    col = await api(`/api/collections/${id}`);
    $('colSel').value = id;
    $('welcome').hidden = true;
    $('people').hidden = true;
    $('person').hidden = false;
    $('personName').textContent = col.person;
    paintStats();
    paintSources();
    paintRunSelect();
    loadCanary();
    if (col.runs.length) await loadRun(col.runs[0].id);
    else { run = null; paintRun(); }
  }

  function paintStats() {
    const parts = [
      [col.documents.length, col.documents.length === 1 ? 'document' : 'documents'],
      [`~${col.est_pages.toLocaleString()}`, 'pages'],
      [col.n_chunks.toLocaleString(), 'chunks'],
      [col.runs.length, col.runs.length === 1 ? 'run' : 'runs'],
    ];
    $('stats').innerHTML = parts.map(([n, l]) => `<span><b>${n}</b> ${l}</span>`).join('');
    $('sourcesCount').textContent = col.documents.length || '';
    $('runBtn').disabled = !col.n_chunks;
    $('runBtn').title = col.n_chunks ? '' : 'Upload sources first';
  }

  $('colSel').addEventListener('change', (e) => { location.hash = e.target.value ? `#/p/${e.target.value}` : '#/'; });

  const dialog = $('personDialog');
  const openPersonDialog = () => { $('personInput').value = ''; dialog.showModal(); $('personInput').focus(); };
  $('newPersonBtn').addEventListener('click', openPersonDialog);
  document.querySelectorAll('[data-new-person]').forEach((b) => b.addEventListener('click', openPersonDialog));
  $('personForm').addEventListener('submit', async (e) => {
    if (e.submitter && e.submitter.value === 'cancel') return;
    e.preventDefault();
    const person = $('personInput').value.trim();
    if (!person) return $('personInput').focus();
    try {
      const c = await api('/api/collections', json({ person }));
      dialog.close();
      people = await api('/api/collections');
      paintPicker();
      location.hash = `#/p/${c.id}`;
      setTab('sources');
    } catch (err) { fail(err); }
  });

  // ---------- tabs ----------

  function setTab(name) {
    document.querySelectorAll('[role=tab]').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
    document.querySelectorAll('[data-panel]').forEach((p) => { p.hidden = p.dataset.panel !== name; });
  }
  document.querySelectorAll('[role=tab]').forEach((t) => t.addEventListener('click', () => setTab(t.dataset.tab)));
  $('goSources').addEventListener('click', () => setTab('sources'));

  // ---------- sources ----------

  function paintSources() {
    const docs = col.documents;
    $('docTable').innerHTML = docs.length ? `
      <thead><tr><th>File</th><th class="num">Words</th><th class="num">Pages</th><th class="num">Chunks</th><th>Doc ID</th></tr></thead>
      <tbody>${docs.map((d) => `<tr><td>${esc(d.filename)}</td><td class="num">${d.n_words.toLocaleString()}</td>
        <td class="num">~${Math.round(d.n_words / 300)}</td><td class="num">${d.n_chunks}</td><td class="mono">${d.id}</td></tr>`).join('')}</tbody>` : '';
    $('settings').textContent = `Chunks of ~350 words with 60 words of overlap · embeddings ${cfg.embed_model} · writer ${cfg.chat_model} · top ${cfg.top_k} chunks per question, kept if similarity ≥ ${cfg.min_similarity}`;
  }

  const fmtSize = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
  function paintQueue() {
    $('queue').hidden = !queue.length;
    $('queueList').innerHTML = queue.map((f) => `<li><span>${esc(f.name)}</span><span>${fmtSize(f.size)}</span></li>`).join('');
    $('upBtn').textContent = `Chunk & embed ${queue.length} file${queue.length === 1 ? '' : 's'}`;
  }
  const addFiles = (list) => { queue = queue.concat([...list]); paintQueue(); };
  $('files').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
  const drop = $('drop');
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));
  $('clearQueue').addEventListener('click', () => { queue = []; paintQueue(); });

  $('upBtn').addEventListener('click', async () => {
    if (!queue.length || !col) return;
    const btn = $('upBtn');
    busy(btn, true, `Embedding ${queue.length} file${queue.length === 1 ? '' : 's'}…`);
    const fd = new FormData();
    queue.forEach((f) => fd.append('files', f));
    try {
      const r = await api(`/api/collections/${col.id}/upload`, { method: 'POST', body: fd });
      const chunks = r.added.reduce((n, a) => n + a.n_chunks, 0);
      toast(`Added ${r.added.length} file${r.added.length === 1 ? '' : 's'} · ${chunks.toLocaleString()} chunks`);
      queue = []; paintQueue();
      col = await api(`/api/collections/${col.id}`);
      paintStats(); paintSources();
    } catch (e) { fail(e); } finally { busy(btn, false); paintQueue(); }
  });

  // ---------- runs ----------

  function paintRunSelect() {
    const sel = $('runSel');
    sel.innerHTML = col.runs.map((r, i) => `<option value="${r.id}">${i === 0 ? 'Latest run' : 'Run'} · ${new Date(r.created_at).toLocaleString()}${r.status !== 'done' ? ` · ${r.status}` : ''}</option>`).join('');
    sel.hidden = col.runs.length < 1;
  }
  $('runSel').addEventListener('change', (e) => loadRun(e.target.value).catch(fail));

  $('runBtn').addEventListener('click', async () => {
    if (!col) return;
    try {
      const r = await api(`/api/collections/${col.id}/runs`, { method: 'POST' });
      col = await api(`/api/collections/${col.id}`);
      paintStats(); paintRunSelect();
      setTab('story');
      await loadRun(r.run_id);
    } catch (e) { fail(e); }
  });

  async function loadRun(id) {
    clearTimeout(pollTimer);
    run = await api(`/api/runs/${id}`);
    $('runSel').value = id;
    paintRun();
    if (run.status === 'running') pollTimer = setTimeout(() => loadRun(id).catch(fail), 1500);
    else if (col.runs.some((r) => r.id === id && r.status !== run.status)) {
      col = await api(`/api/collections/${col.id}`); paintRunSelect(); paintStats(); $('runSel').value = id;
    }
  }

  function paintProgress() {
    const card = $('progressCard');
    const running = run && run.status === 'running';
    card.hidden = !running;
    $('runBtn').disabled = running || !col.n_chunks;
    if (!running) return;
    const p = run.progress || 0;
    const stage = p < 20 ? 0 : p === 20 ? 1 : 2;
    card.querySelectorAll('li').forEach((li) => {
      const s = Number(li.dataset.step);
      li.className = s < stage ? 'done' : s === stage ? 'active' : '';
    });
    $('stepQ').textContent = `${Math.min(p, 20)} / 20 questions`;
    $('bar').style.width = `${Math.round((p / 22) * 100)}%`;
  }

  function paintRun() {
    paintProgress();
    paintStory();
    paintAnswers();
    paintAudit();
    $('logLinks').hidden = !run || run.status === 'running';
  }

  // Downloads go through fetch so the password header is sent.
  async function download(path) {
    const res = await fetch(path, { headers: { 'X-App-Password': password } });
    if (!res.ok) throw new ApiError(res.status, '', 'Download failed.');
    const name = (res.headers.get('Content-Disposition') || '').match(/filename="([^"]+)"/)?.[1] || 'download';
    const url = URL.createObjectURL(await res.blob());
    const a = Object.assign(document.createElement('a'), { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  document.querySelectorAll('[data-download]').forEach((b) => b.addEventListener('click', () => {
    if (run) download(`/api/runs/${run.id}/report.${b.dataset.download}`).catch(fail);
  }));

  // ---------- story ----------

  const CITE = /\[([A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*)\]/g;
  const splitSentences = (para) => para.split(/(?<=[.!?\]])\s+(?=[A-Z"“‘(])/).filter((s) => s.trim());

  function citeChips(text, numbers) {
    return esc(text).replace(CITE, (_, ids) => ids.split(',').map((i) => i.trim()).map((i) => {
      if (numbers) {
        if (!numbers.has(i)) numbers.set(i, numbers.size + 1);
        return `<button class="cite" data-chunk="${i}" title="${i}">${numbers.get(i)}</button>`;
      }
      return `<button class="cite inline" data-chunk="${i}">${i}</button>`;
    }).join(''));
  }

  function paintStory() {
    const box = $('story');
    if (!run) {
      box.innerHTML = col.n_chunks
        ? `<div class="empty"><strong>No story yet</strong>Run the 20 questions against ${esc(col.person)}’s sources to write one.<br><button class="btn primary" type="button" data-run>Generate story</button></div>`
        : `<div class="empty"><strong>Start with sources</strong>Upload pages about ${esc(col.person)}, then generate a story.<br><button class="btn primary" type="button" data-sources>Add sources</button></div>`;
      box.querySelector('[data-run]')?.addEventListener('click', () => $('runBtn').click());
      box.querySelector('[data-sources]')?.addEventListener('click', () => setTab('sources'));
      return;
    }
    if (run.status === 'error') { box.innerHTML = `<div class="empty"><strong>This run failed</strong>${esc(run.error)}</div>`; return; }
    if (!run.story) { box.innerHTML = `<div class="empty"><strong>Writing…</strong>The story appears once all 20 questions are answered.</div>`; return; }

    const audit = run.audit ? run.audit.sentences : [];
    const numbers = new Map();
    let n = 0;
    const paras = run.story.split(/\n\s*\n/).map((p) => p.split(/\s+/).join(' ')).filter(Boolean);
    const html = paras.map((p) => `<p>${splitSentences(p).map((s) => {
      const a = audit[n++];
      const flag = a && a.flag;
      return `<span class="sent${flag ? ' flagged' : ''}"${flag ? ` title="${esc(a.flag)}${a.outside_claims ? ': ' + esc(a.outside_claims) : ''}"` : ''}>${citeChips(s, numbers)}</span>`;
    }).join(' ')}</p>`).join('');
    const flagged = run.audit ? run.audit.flagged : null;
    box.innerHTML = `<div class="story-body">${html}</div>
      <div class="story-foot"><span>${n} sentences</span><span>${numbers.size} chunks cited</span>
      ${flagged === null ? '<span>Audit running…</span>' : `<span>${flagged ? `<b style="color:var(--bad)">${flagged} flagged</b>` : 'No sentence flagged'}</span>`}
      <span>Click a number to read the chunk</span></div>`;
    box.classList.toggle('hl', $('highlight').checked);
  }
  $('highlight').addEventListener('change', (e) => $('story').classList.toggle('hl', e.target.checked));

  // ---------- answers ----------

  function paintAnswers() {
    const list = run ? run.answers : [];
    const found = list.filter((a) => a.answer !== NOT_FOUND).length;
    $('answersCount').textContent = list.length ? `${found}/${list.length}` : '';
    if (!list.length) {
      $('answers').innerHTML = `<div class="empty"><strong>No answers yet</strong>${run ? 'Answers appear as each question finishes.' : 'Generate a story to see each question’s retrieval.'}</div>`;
      return;
    }
    const open = new Set([...document.querySelectorAll('.answer[open]')].map((d) => d.dataset.q));
    $('answers').innerHTML = list.map((a) => {
      const nf = a.answer === NOT_FOUND;
      const top = Math.max(...a.retrieved.map((h) => h.score), 0.0001);
      return `<details class="answer" data-q="${a.qnum}"${open.has(String(a.qnum)) ? ' open' : ''}>
        <summary><span class="qnum">Q${String(a.qnum).padStart(2, '0')}</span><span class="qtext">${esc(a.question)}</span>
          <span class="badge ${nf ? 'none' : 'good'}">${nf ? 'Not in the sources' : 'Answered'}</span></summary>
        <div class="answer-body">
          <p class="answer-text${nf ? ' nf' : ''}">${nf ? 'Not in the sources.' : citeChips(a.answer)}</p>
          <p class="sub-label">Retrieved chunks</p>
          ${a.retrieved.length ? `<div class="hits">${a.retrieved.map((h) => `<div class="hit">
              <button class="cite inline" data-chunk="${h.chunk_id}">${h.chunk_id}</button>
              <div class="score"><div style="width:${Math.round((h.score / top) * 100)}%"></div></div>
              <span class="score-n">${h.score.toFixed(3)}</span></div>`).join('')}</div>`
            : `<p class="muted" style="font-size:14px;margin:0">Nothing scored above ${cfg.min_similarity}, so no model call was made.</p>`}
          <details class="prompt-toggle"><summary>Prompt sent</summary><pre>${esc(a.prompt)}</pre></details>
        </div></details>`;
    }).join('');
    const waiting = 20 - list.length;
    if (run.status === 'running' && waiting > 0) $('answers').insertAdjacentHTML('beforeend', `<p class="muted" style="font-size:14px">${waiting} more on the way…</p>`);
  }

  // ---------- audit ----------

  function paintAudit() {
    const a = run && run.audit;
    $('auditCount').textContent = a ? (a.flagged || '✓') : '';
    $('auditCount').classList.toggle('bad', !!(a && a.flagged));
    if (!a) {
      $('tiles').innerHTML = `<div class="empty" style="grid-column:1/-1"><strong>No audit yet</strong>${run ? 'The audit runs after the story is written.' : 'Generate a story first.'}</div>`;
      $('auditTable').innerHTML = '';
      return;
    }
    const traced = a.total - a.flagged;
    $('tiles').innerHTML = `
      <div class="tile"><b>${a.total}</b><span>sentences in the story</span></div>
      <div class="tile good"><b>${traced}</b><span>traced to a chunk</span></div>
      <div class="tile ${a.flagged ? 'bad' : 'good'}"><b>${a.flagged}</b><span>flagged as possible training data</span></div>
      <div class="tile"><b>${a.questions_not_in_sources.length}</b><span>questions not in the sources${a.questions_not_in_sources.length ? ` (Q${a.questions_not_in_sources.join(', Q')})` : ''}</span></div>`;
    const label = (s) => {
      if (s.flag === 'no citation') return '<span class="badge bad">No citation</span>';
      if (s.invalid_citations.length) return '<span class="badge bad">Not retrieved</span>';
      if (s.verdict === 'unsupported') return '<span class="badge bad">Unsupported</span>';
      if (s.verdict === 'partial') return '<span class="badge warn">Partly supported</span>';
      return '<span class="badge good">Supported</span>';
    };
    $('auditTable').innerHTML = `<thead><tr><th>#</th><th>Sentence</th><th>Chunks</th><th>Check</th></tr></thead><tbody>${a.sentences.map((s) => `
      <tr class="${s.flag ? 'flag-row' : ''}"><td class="mono">${s.n}</td>
        <td class="sentence">${esc(s.sentence.replace(/\s*\[[^\]]+\]/g, ''))}</td>
        <td>${s.citations.map((c) => `<button class="cite inline" data-chunk="${c}">${c}</button>`).join(' ') || '<span class="muted">none</span>'}</td>
        <td>${label(s)}${s.outside_claims ? `<span class="why">Not in chunk: ${esc(s.outside_claims)}</span>` : ''}</td></tr>`).join('')}</tbody>`;
  }

  // ---------- canary ----------

  async function loadCanary() {
    try { paintCanary(await api(`/api/collections/${col.id}/canary-logs`)); } catch (e) { fail(e); }
  }
  function paintCanary(rows) {
    $('canaryList').innerHTML = rows.length ? rows.map((r) => {
      const nf = r.response === NOT_FOUND;
      return `<div class="canary"><div class="canary-q"><span>${esc(r.question)}</span>
          <span class="badge ${nf ? 'good' : 'warn'}">${nf ? 'Not in the sources' : 'Answered from sources'}</span></div>
        <p class="canary-a${nf ? ' nf' : ''}">${nf ? 'Not in the sources.' : citeChips(r.response)}</p>
        <div class="canary-meta">${r.retrieved.length ? 'Retrieved ' + r.retrieved.map((h) => `<button class="cite inline" data-chunk="${h.chunk_id}">${h.chunk_id}</button>`).join(' ') : `Nothing scored above ${cfg.min_similarity}`}
          · ${new Date(r.created_at).toLocaleString()}</div></div>`;
    }).join('') : '<div class="empty"><strong>No canary questions yet</strong>Try one fact you know is true but left out of the corpus.</div>';
  }
  $('askForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = $('canaryQ').value.trim();
    if (!q || !col) return;
    const btn = $('askBtn');
    busy(btn, true, 'Asking…');
    try { await api(`/api/collections/${col.id}/ask`, json({ question: q })); $('canaryQ').value = ''; await loadCanary(); }
    catch (err) { fail(err); } finally { busy(btn, false); }
  });

  // ---------- chunk drawer ----------

  const drawer = $('drawer'), scrim = $('scrim');
  async function openChunk(id) {
    $('drawerTitle').textContent = id;
    $('drawerMeta').textContent = 'Loading…';
    $('drawerBody').textContent = '';
    drawer.classList.add('open'); drawer.setAttribute('aria-hidden', 'false'); scrim.hidden = false;
    try {
      const c = await api(`/api/chunks/${id}`);
      $('drawerMeta').textContent = `${c.filename} · chunk ${c.ord + 1}`;
      $('drawerBody').innerHTML = `<p>${esc(c.text)}</p>`;
    } catch (e) { $('drawerMeta').textContent = 'Not found'; $('drawerBody').textContent = e.message; }
  }
  function closeChunk() { drawer.classList.remove('open'); drawer.setAttribute('aria-hidden', 'true'); scrim.hidden = true; }
  document.addEventListener('click', (e) => {
    const c = e.target.closest('[data-chunk]');
    if (c) { e.preventDefault(); e.stopPropagation(); openChunk(c.dataset.chunk); }
  });
  $('drawerClose').addEventListener('click', closeChunk);
  scrim.addEventListener('click', closeChunk);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeChunk(); });

  // ---------- on load ----------

  (async () => {
    try { cfg = await (await fetch('/api/config')).json(); } catch {}
    const saved = store.get(PASSWORD_KEY);
    if (!saved) { pwField.focus({ preventScroll: true }); return; }
    // Already know the password on this device: check it quietly and skip the gate.
    gate.hidden = true;
    const showGate = () => { store.drop(PASSWORD_KEY); document.documentElement.classList.remove('known-word'); gate.hidden = false; pwField.focus(); };
    try {
      if ((await tryPassword(saved)) === true) { password = saved; openApp(); } else showGate();
    } catch { showGate(); }
  })();
})();
