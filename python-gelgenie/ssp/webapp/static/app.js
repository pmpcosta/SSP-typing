/* SSP Typing local UI — vanilla JS, no build step.
 * Flow: the image is staged first (POST /api/uploads); a joined strip goes
 * straight to typing, a raw gel photo goes through the layout step
 * (layout.js) and the session is created from the confirmed layout.
 * Interactions: click band = delete/restore, shift+click = add band,
 * click lane header = cycle force positive/negative/auto,
 * match-tol slider re-types live. All edits round-trip through
 * POST /api/sessions/{sid}/edits which recomputes everything server-side.
 */
'use strict';

const HEADER_H = 26;               // canvas px reserved for lane headers
const KIND_COLORS = {
  product:     { stroke: '#2ecc71', fill: 'rgba(46,204,113,0.25)' },
  control:     { stroke: '#4da3ff', fill: 'rgba(77,163,255,0.10)' },
  dimer:       { stroke: '#f1c40f', fill: 'rgba(241,196,15,0.15)' },
  hmw:         { stroke: '#e056fd', fill: 'rgba(224,86,253,0.15)' },
  unexplained: { stroke: '#f39c12', fill: 'rgba(243,156,18,0.20)' },
  unclassified:{ stroke: '#8a93a5', fill: 'rgba(138,147,165,0.12)' },
};

const state = {
  sid: null, data: null, edits: null, img: null, maskCanvas: null,
  scale: null, showMask: false, laneBounds: [],
};

const clone = (o) => JSON.parse(JSON.stringify(o));
let tolTouched = false;   // slider moved since last server state?

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const ct = res.headers.get('content-type') || '';
  const body = ct.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = (body && body.detail) || body || res.statusText;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return body;
}

/* ---------------- init ---------------- */

async function init() {
  await Promise.all([loadTrays(), loadSessions()]);
  bindUI();
  LayoutEditor.bind();
}

async function loadTrays() {
  try {
    const trays = await api('/api/trays');
    const sel = $('tray-select');
    const cur = sel.value;
    sel.innerHTML = '<option value="">— upload a file above —</option>';
    for (const t of trays) {
      const o = document.createElement('option');
      o.value = t.name; o.textContent = t.name;
      sel.appendChild(o);
    }
    if (cur) sel.value = cur;
  } catch (e) { console.warn('tray list failed', e); }
}

async function loadSessions() {
  try {
    const sessions = await api('/api/sessions');
    const ul = $('session-list');
    ul.innerHTML = '';
    for (const s of sessions) {
      const li = document.createElement('li');
      li.dataset.sid = s.id;
      li.innerHTML = `<div>${escapeHtml(s.image_name || s.id)}</div>
        <div class="when">${escapeHtml(s.tray_name || '')} · ${escapeHtml(s.created || '')}</div>`;
      if (s.id === state.sid) li.classList.add('active');
      li.addEventListener('click', () => openSession(s.id));
      ul.appendChild(li);
    }
  } catch (e) { console.warn('session list failed', e); }
}

function bindUI() {
  $('upload-form').addEventListener('submit', runPipeline);
  $('tray-file').addEventListener('change', () => {
    if ($('tray-file').files.length) $('tray-select').value = '';
  });
  $('tray-select').addEventListener('change', () => {
    if ($('tray-select').value) $('tray-file').value = '';
  });
  $('upload-tray-btn').addEventListener('click', () => $('tray-upload').click());
  $('tray-upload').addEventListener('change', uploadTray);

  $('zoom-in').addEventListener('click', () => zoomBy(1.25));
  $('zoom-out').addEventListener('click', () => zoomBy(0.8));
  $('zoom-fit').addEventListener('click', () => { state.scale = null; draw(); });
  $('show-mask').addEventListener('change', async (ev) => {
    state.showMask = ev.target.checked;
    if (state.showMask && !state.maskCanvas && state.sid) await loadMask();
    draw();
  });
  $('reset-edits').addEventListener('click', resetEdits);
  $('relayout-btn').addEventListener('click', relayout);

  const tol = $('match-tol');
  let tolTimer = null;
  tol.addEventListener('input', () => {
    tolTouched = true;
    $('match-tol-val').textContent = Number(tol.value).toFixed(2);
    clearTimeout(tolTimer);
    tolTimer = setTimeout(() => { if (state.sid) queuePush(); }, 450);
  });

  const canvas = $('gel-canvas');
  canvas.addEventListener('click', onCanvasClick);
  canvas.addEventListener('wheel', (ev) => {
    if (!ev.ctrlKey) return;
    ev.preventDefault();
    zoomBy(ev.deltaY < 0 ? 1.1 : 0.9);
  }, { passive: false });
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ---------------- session lifecycle ---------------- */

function trayChoice() {
  const trayFile = $('tray-file').files[0];
  const trayName = $('tray-select').value;
  if (!trayFile && !trayName) throw new Error('Choose a tray file or a stored tray.');
  return { trayFile, trayName };
}

async function createSession(fields) {
  const { trayFile, trayName } = trayChoice();
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) fd.append(k, v);
  if (trayFile) fd.append('tray', trayFile);
  else fd.append('tray_name', trayName);
  fd.append('min_area', String($('min-area').value || 5));
  if ($('invert-flag').checked) fd.append('invert', 'true');
  const data = await api('/api/sessions', { method: 'POST', body: fd });
  await setState(data, { refit: true, newSession: true });
  await Promise.all([loadSessions(), loadTrays()]);
  return data;
}

function openLayoutStep(info, opts = {}) {
  return LayoutEditor.open(info, {
    ...opts,
    api,
    onConfirm: (uid, layout) =>
      createSession({ upload_id: uid, layout: JSON.stringify(layout) }),
    onCancel: () => {
      if (state.data) $('workspace').hidden = false;
      else $('placeholder').hidden = false;
    },
  });
}

async function runPipeline(ev) {
  ev.preventDefault();
  const imgFile = $('image-file').files[0];
  const status = $('upload-status');
  if (!imgFile) return;
  try {
    trayChoice();
  } catch (e) {
    status.className = 'status error';
    status.textContent = e.message;
    return;
  }
  const btn = $('run-btn');
  btn.disabled = true;
  status.className = 'status';
  status.textContent = 'uploading…';
  try {
    const fd = new FormData();
    fd.append('image', imgFile);
    const info = await api('/api/uploads', { method: 'POST', body: fd });
    if (info.prestitched) {
      status.textContent = 'segmenting + typing… (a few seconds)';
      await createSession({ upload_id: info.upload_id });
    } else {
      status.textContent = '';
      await openLayoutStep(info);
    }
    status.textContent = '';
  } catch (e) {
    status.className = 'status error';
    status.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

async function relayout() {
  if (!state.sid) return;
  try {
    const info = await api(`/api/sessions/${state.sid}/relayout`, { method: 'POST' });
    // re-type against the same stored tray
    if (info.tray_store_name) {
      $('tray-file').value = '';
      $('tray-select').value = info.tray_store_name;
    }
    await openLayoutStep(info, { layout: info.layout });
  } catch (e) {
    $('upload-status').className = 'status error';
    $('upload-status').textContent = e.message;
  }
}

async function openSession(sid) {
  try {
    LayoutEditor.close();
    const data = await api(`/api/sessions/${sid}`);
    await setState(data, { refit: true, newSession: true });
    loadSessions();
  } catch (e) {
    $('upload-status').className = 'status error';
    $('upload-status').textContent = e.message;
  }
}

async function uploadTray(ev) {
  const f = ev.target.files[0];
  ev.target.value = '';
  if (!f) return;
  const fd = new FormData();
  fd.append('tray', f);
  const status = $('upload-status');
  try {
    const r = await api('/api/trays', { method: 'POST', body: fd });
    status.className = 'status';
    status.textContent = `stored tray ${r.name} (${r.lanes} lanes)`;
    await loadTrays();
  } catch (e) {
    status.className = 'status error';
    status.textContent = e.message;
  }
}

async function setState(data, { refit = false, newSession = false,
                                adoptEdits = true } = {}) {
  if (newSession || data.id !== state.sid) {
    state.sid = data.id;
    state.img = null;
    state.maskCanvas = null;
    state.showMask = false;
    $('show-mask').checked = false;
    if (refit) state.scale = null;
  }
  state.data = data;
  if (adoptEdits) {
    state.edits = clone(data.edits);
    // snap the slider only on fresh loads; inside the push-resend loop keep
    // the user's in-flight slider input (tolTouched) so it is re-sent
    // instead of silently dropped
    const tol = $('match-tol');
    tol.value = String(data.params.match_tol);
    $('match-tol-val').textContent = Number(data.params.match_tol).toFixed(2);
    tolTouched = false;
  }

  if (!state.img) {
    state.img = new Image();
    state.img.src = `/api/sessions/${data.id}/image`;
    await state.img.decode();
  }
  if (state.showMask && !state.maskCanvas) await loadMask();

  $('placeholder').hidden = true;
  $('layout-step').hidden = true;
  $('workspace').hidden = false;
  $('relayout-btn').hidden = !data.has_original;
  $('lane-source').textContent = data.lane_assignment === 'layout'
    ? 'lanes from confirmed layout' : '';
  $('session-info').textContent =
    `${data.image_name} × ${data.tray_name}  (${data.id})`;
  $('dl-typing').href = `/api/sessions/${data.id}/typing.json`;
  $('dl-report').href = `/api/sessions/${data.id}/report.txt`;

  renderWarnings();
  renderTyping();
  renderLaneTable();
  draw();
}

async function loadMask() {
  try {
    const img = new Image();
    img.src = `/api/sessions/${state.sid}/mask`;
    await img.decode();
    const c = document.createElement('canvas');
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < d.data.length; i += 4) {
      const on = d.data[i] > 127;
      d.data[i] = 46; d.data[i + 1] = 204; d.data[i + 2] = 113;
      d.data[i + 3] = on ? 110 : 0;
    }
    ctx.putImageData(d, 0, 0);
    state.maskCanvas = c;
  } catch (e) {
    console.warn('mask load failed', e);
    state.showMask = false;
    $('show-mask').checked = false;
  }
}

/* ---------------- edits ---------------- */

let pushInFlight = false;
let pushQueued = false;

function currentEdits() {
  // client-authoritative edit set; the server copy may lag during rapid edits
  return state.edits || { lane_overrides: [], deleted_bands: [], added_bands: [] };
}

// One POST in flight at a time; interactions during a push mark the queue and
// the LATEST state.edits is re-sent afterwards, so rapid clicks can never
// overwrite each other server-side (lost-update race).
async function queuePush() {
  if (!state.sid) return;
  if (pushInFlight) { pushQueued = true; return; }
  pushInFlight = true;
  try {
    do {
      pushQueued = false;
      const body = clone(currentEdits());
      if (tolTouched) body.match_tol = Number($('match-tol').value);
      const data = await api(`/api/sessions/${state.sid}/edits`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      await setState(data, { adoptEdits: false });
    } while (pushQueued);
  } catch (e) {
    $('upload-status').className = 'status error';
    $('upload-status').textContent = e.message;
    if (state.data) {         // roll back to the last server-confirmed edits
      state.edits = clone(state.data.edits);
      renderTyping(); renderLaneTable(); draw();
    }
  } finally {
    pushInFlight = false;
  }
}

async function resetEdits() {
  const d = (state.data && state.data.defaults) || {};
  const tol = d.match_tol ?? 0.25;
  $('match-tol').value = String(tol);
  $('match-tol-val').textContent = Number(tol).toFixed(2);
  tolTouched = true;   // restore the server default explicitly
  state.edits = { lane_overrides: [], deleted_bands: [], added_bands: [] };
  await queuePush();
}

/* ---------------- canvas ---------------- */

function fixedCenters(lanes) {
  // lanes inserted for empty positions can have center_x == 0; interpolate
  const cs = lanes.map((l) => l.center_x);
  const known = cs.filter((c) => c > 0);
  let pitch = 40;
  if (known.length > 1) {
    const diffs = known.slice(1).map((v, i) => v - known[i]).filter((d) => d > 0);
    if (diffs.length) {
      diffs.sort((a, b) => a - b);
      pitch = diffs[Math.floor(diffs.length / 2)];
    }
  }
  for (let i = 0; i < cs.length; i++) {
    if (cs[i] > 0) continue;
    let v = null;
    for (let j = i - 1; j >= 0; j--) if (cs[j] > 0) { v = cs[j] + (i - j) * pitch; break; }
    if (v === null) for (let j = i + 1; j < cs.length; j++)
      if (cs[j] > 0) { v = cs[j] - (j - i) * pitch; break; }
    cs[i] = v === null ? i * pitch : v;
  }
  return cs;
}

function computeLaneBounds() {
  const lanes = state.data.analysis.lanes;
  const cs = fixedCenters(lanes);
  const w = state.img.naturalWidth;
  const bounds = [];
  for (let i = 0; i < lanes.length; i++) {
    const left = i === 0 ? 0 : (cs[i - 1] + cs[i]) / 2;
    const right = i === lanes.length - 1 ? w : (cs[i] + cs[i + 1]) / 2;
    bounds.push({ mix: lanes[i].mix, left, right, center: cs[i] });
  }
  state.laneBounds = bounds;
  return bounds;
}

function zoomBy(f) {
  state.scale = clampScale((state.scale || fitScale()) * f);
  draw();
}
function fitScale() {
  const cont = $('gel-container');
  return Math.max(0.05, (cont.clientWidth - 24) / state.img.naturalWidth);
}
function clampScale(s) { return Math.min(3, Math.max(0.05, s)); }

function draw() {
  if (!state.data || !state.img) return;
  const canvas = $('gel-canvas');
  const ctx = canvas.getContext('2d');
  const imgW = state.img.naturalWidth, imgH = state.img.naturalHeight;
  if (state.scale === null) state.scale = fitScale();
  const s = state.scale;
  canvas.width = Math.round(imgW * s);
  canvas.height = Math.round(imgH * s) + HEADER_H;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const bounds = computeLaneBounds();
  const lanes = state.data.analysis.lanes;
  const overrides = {};
  for (const o of state.data.edits.lane_overrides || []) overrides[o.mix] = o;

  // image + optional mask
  ctx.drawImage(state.img, 0, HEADER_H, imgW * s, imgH * s);
  if (state.showMask && state.maskCanvas) {
    ctx.drawImage(state.maskCanvas, 0, HEADER_H, imgW * s, imgH * s);
  }

  // lane headers + dividers
  for (let i = 0; i < lanes.length; i++) {
    const lane = lanes[i], b = bounds[i];
    const x0 = b.left * s, x1 = b.right * s;
    let bg = '#2a2f3a';
    if (!lane.valid) bg = '#5d2a24';
    else if (lane.positive) bg = '#1e5c38';
    ctx.fillStyle = bg;
    ctx.fillRect(x0 + 0.5, 0, Math.max(1, x1 - x0 - 1), HEADER_H - 2);
    if (overrides[lane.mix]) {
      ctx.strokeStyle = '#4da3ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 + 1, 1, Math.max(1, x1 - x0 - 2), HEADER_H - 4);
      ctx.lineWidth = 1;
    }
    if (x1 - x0 > 18) {
      ctx.fillStyle = '#d8dee9';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(String(lane.mix) + (overrides[lane.mix] ? '*' : ''),
                   (x0 + x1) / 2, HEADER_H - 9);
    }
    if (i > 0) {
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x0, HEADER_H); ctx.lineTo(x0, HEADER_H + imgH * s);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // bands from the (re-computed) analysis
  for (const lane of lanes) {
    for (const bd of lane.bands || []) drawBand(ctx, bd, s, false);
  }
  // deleted raw bands shown struck through
  const deleted = new Set(state.data.edits.deleted_bands || []);
  if (deleted.size) {
    for (const bd of state.data.bands || []) {
      if (deleted.has(bd.id)) drawBand(ctx, bd, s, true);
    }
  }
}

function drawBand(ctx, bd, s, isDeleted) {
  const [y0, x0, y1, x1] = bd.bbox;
  let px = x0 * s, py = HEADER_H + y0 * s;
  let pw = Math.max((x1 - x0) * s, 3), ph = Math.max((y1 - y0) * s, 3);
  if (bd.synthetic) {           // 1-px synthetic bands: draw a visible marker
    px = bd.centroid_x * s - 5; py = HEADER_H + bd.centroid_y * s - 5;
    pw = 10; ph = 10;
  }
  const col = KIND_COLORS[bd.kind] || KIND_COLORS.unclassified;
  if (isDeleted) {
    ctx.strokeStyle = '#ff5252';
    ctx.strokeRect(px - 1, py - 1, pw + 2, ph + 2);
    ctx.beginPath();
    ctx.moveTo(px - 1, py - 1); ctx.lineTo(px + pw + 1, py + ph + 1);
    ctx.moveTo(px + pw + 1, py - 1); ctx.lineTo(px - 1, py + ph + 1);
    ctx.stroke();
    return;
  }
  ctx.fillStyle = col.fill;
  ctx.fillRect(px, py, pw, ph);
  ctx.strokeStyle = col.stroke;
  if (bd.synthetic) ctx.setLineDash([3, 2]);
  ctx.strokeRect(px, py, pw, ph);
  ctx.setLineDash([]);
}

function onCanvasClick(ev) {
  if (!state.data) return;
  const rect = ev.target.getBoundingClientRect();
  const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
  const s = state.scale;

  if (cy < HEADER_H) {          // lane header: cycle override
    const ix = cx / s;
    const b = state.laneBounds.find((bb) => ix >= bb.left && ix < bb.right);
    if (b) cycleOverride(b.mix);
    return;
  }
  const ix = cx / s, iy = (cy - HEADER_H) / s;
  if (iy < 0) return;

  if (ev.shiftKey) {            // add band
    currentEdits().added_bands.push(
      { x: Math.round(ix * 10) / 10, y: Math.round(iy * 10) / 10 });
    queuePush();
    return;
  }
  // hit-test bands of the lane under the cursor only: at fit-zoom the 3-px
  // pad reaches ~35 image px (wider than a lane), so without the lane
  // restriction a click can grab the neighboring lane's band. Real bands
  // first (synthetic area=1 would otherwise sort first and shadow them).
  const lb = state.laneBounds.find((bb) => ix >= bb.left && ix < bb.right);
  const hits = [];
  for (const lane of state.data.analysis.lanes) {
    if (lb && lane.mix !== lb.mix) continue;
    for (const bd of lane.bands || []) {
      const [y0, x0, y1, x1] = bd.bbox;
      const pad = 3 / s;
      if (ix >= x0 - pad && ix <= x1 + pad && iy >= y0 - pad && iy <= y1 + pad) {
        hits.push(bd);
      }
    }
  }
  if (hits.length) {
    hits.sort((a, b) => ((a.synthetic ? 1 : 0) - (b.synthetic ? 1 : 0))
                        || (a.area - b.area));
    const hit = hits[0];
    const edits = currentEdits();
    if (hit.synthetic) {
      // remove a user-added band by matching its stored position
      const i = edits.added_bands.findIndex((ab) =>
        Math.abs(ab.x - hit.centroid_x) < 0.51 &&
        Math.abs(ab.y - hit.centroid_y) < 0.51);
      if (i !== -1) edits.added_bands.splice(i, 1);
      else return;
    } else if (!edits.deleted_bands.includes(hit.id)) {
      edits.deleted_bands.push(hit.id);
    } else {
      return;   // already deleted client-side; don't double-push
    }
    queuePush();
    return;
  }
  // hit-test deleted raw bands -> restore (same lane restriction as deletes:
  // raw bands carry no lane field, so match their centroid against the bounds)
  const deleted = new Set((state.data.edits.deleted_bands || []));
  for (const bd of state.data.bands || []) {
    if (!deleted.has(bd.id)) continue;
    if (lb) {
      const bl = state.laneBounds.find(
        (bb) => bd.centroid_x >= bb.left && bd.centroid_x < bb.right);
      if (!bl || bl.mix !== lb.mix) continue;
    }
    const [y0, x0, y1, x1] = bd.bbox;
    const pad = 3 / s;
    if (ix >= x0 - pad && ix <= x1 + pad && iy >= y0 - pad && iy <= y1 + pad) {
      const edits = currentEdits();
      edits.deleted_bands = edits.deleted_bands.filter((d) => d !== bd.id);
      queuePush();
      return;
    }
  }
}

function cycleOverride(mix) {
  const edits = currentEdits();
  const lane = state.data.analysis.lanes.find((l) => l.mix === mix);
  const i = edits.lane_overrides.findIndex((o) => o.mix === mix);
  if (i === -1) {
    const ov = { mix, positive: true };
    if (lane && !lane.valid) ov.valid = true;   // force-positive needs validity
    edits.lane_overrides.push(ov);
  } else if (edits.lane_overrides[i].positive === true) {
    const next = { mix, positive: false };
    // carry a forced-valid flag so stage 2 stays meaningful on invalid lanes
    if (edits.lane_overrides[i].valid !== undefined) {
      next.valid = edits.lane_overrides[i].valid;
    }
    edits.lane_overrides[i] = next;
  } else {
    edits.lane_overrides.splice(i, 1);          // back to auto
  }
  queuePush();
}

/* ---------------- results panels ---------------- */

function genotypeText(loc) {
  if (!loc.best_pairs || !loc.best_pairs.length) return 'no call';
  if (loc.allele_options) {
    return loc.allele_options.map((opts) => opts.join('/')).join('  +  ');
  }
  const shown = loc.best_pairs.slice(0, 6).map((p) => `${p[0]} + ${p[1]}`);
  let t = shown.join(';  ');
  if (loc.best_pairs.length > 6) t += `;  …(+${loc.best_pairs.length - 6} more)`;
  return t;
}

function renderWarnings() {
  const ws = [...(state.data.analysis.warnings || []),
              ...(state.data.typing.warnings || [])];
  $('global-warnings').textContent = [...new Set(ws)].join(' · ');
}

function renderTyping() {
  const box = $('loci-results');
  box.innerHTML = '';
  for (const [locus, loc] of Object.entries(state.data.typing.loci)) {
    const card = document.createElement('div');
    const noCall = !loc.best_pairs || !loc.best_pairs.length;
    card.className = 'locus-card' + (noCall ? ' nocall'
                     : (loc.score > 0 || (loc.warnings || []).length ? ' warn' : ''));
    const inv = (loc.invalid || []).length ? ` · invalid lanes: ${loc.invalid.join(', ')}` : '';
    card.innerHTML = `
      <div class="geno">HLA-${escapeHtml(locus)}: ${escapeHtml(genotypeText(loc))}</div>
      <div class="meta">score ${loc.score ?? '–'} · positives:
        ${loc.positives.length ? loc.positives.join(', ') : '—'}${escapeHtml(inv)}</div>
      ${(loc.warnings || []).map((w) => `<div class="lwarn">⚠ ${escapeHtml(w)}</div>`).join('')}`;
    box.appendChild(card);
  }
}

function renderLaneTable() {
  const tbody = $('lane-table').querySelector('tbody');
  tbody.innerHTML = '';
  const lanesByMix = {};
  for (const l of state.data.analysis.lanes) lanesByMix[l.mix] = l;
  const overrides = {};
  for (const o of state.data.edits.lane_overrides || []) overrides[o.mix] = o;

  for (const tl of state.data.tray.lanes) {
    const lane = lanesByMix[tl.mix] || {};
    const tr = document.createElement('tr');
    const anomalous = (lane.bands || []).some((b) => b.kind === 'unexplained' || b.kind === 'hmw');
    if (!lane.valid) tr.className = 'invalid';
    else if (lane.positive) tr.className = 'positive';
    else if (anomalous) tr.className = 'anomalous';

    const call = !lane.valid ? 'INVALID' : (lane.positive ? 'POSITIVE' : 'negative');
    const ov = overrides[tl.mix];
    const ovTxt = ov ? ` <span class="ov">(forced ${ov.positive ? '+' : '−'})</span>` : '';
    const bands = (lane.bands || [])
      .filter((b) => b.kind !== 'control')
      .map((b) => `${b.kind}${b.synthetic ? '(added)' : ''} @ ${b.bp_est ?? '?'}bp`
           + (b.matched_bp ? ` ~${b.matched_bp}` : ''))
      .join('; ');
    tr.innerHTML = `
      <td>${tl.mix}</td>
      <td>${escapeHtml(tl.locus)}</td>
      <td class="spec" title="${escapeHtml(tl.specificities)}">${escapeHtml(tl.specificities || '')}</td>
      <td>${lane.valid ? '✓' : '✗'}</td>
      <td class="call">${call}${ovTxt}</td>
      <td>${escapeHtml(bands)}</td>`;
    tbody.appendChild(tr);
  }
}

init();
