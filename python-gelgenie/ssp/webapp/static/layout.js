/* SSP Typing — gel layout step (vanilla JS, no build step).
 *
 * Shows the raw gel photo with the suggested layout: one outline per gel
 * piece (drag its 4 corners), row cut lines (drag up/down onto the top of
 * the wells), and used-lane bounds (drag the ◂ ▸ handles to exclude empty
 * wells; shift = this row only). Mix numbers are drawn live on every lane
 * so a mis-numbered edge is visible before typing. The joined strip is
 * previewed from the server after each change. Geometry mirrors
 * ssp/layout.py: rectified block size = mean opposite edge lengths, rows as
 * fractions of the rectified block, projective map rectified -> image.
 */
'use strict';

const LayoutEditor = (() => {
  const COLORS = ['#00c8ff', '#ffaa00', '#3cdc3c', '#ff46c8', '#ffff3c', '#a078ff'];
  const HANDLE_R = 7;
  const MIN_GAP = 0.02;          // min row height / bound gap (fraction)
  const ed = {
    uid: null, info: null, layout: null, img: null, scale: null,
    drag: null, previewTimer: null, previewUrl: null, previewSeq: 0,
    onConfirm: null, onCancel: null,
    gen: 0,   // editor instance: bumped by open() / close(); async work that
              // started under another instance must not touch this one
  };
  const $ = (id) => document.getElementById(id);

  /* ---------- geometry (mirrors ssp/layout.py) ---------- */

  function blockSize(quad) {
    const d = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
    const [tl, tr, br, bl] = quad;
    return [Math.max(1, Math.round((d(tl, tr) + d(bl, br)) / 2)),
            Math.max(1, Math.round((d(tl, bl) + d(tr, br)) / 2))];
  }

  function solve(A, b) {           // Gaussian elimination with pivoting
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-12) return null;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map((row, i) => row[n] / row[i]);
  }

  function homography(quad) {      // rectified (x, y) -> image (x, y)
    const [w, h] = blockSize(quad);
    const src = [[0, 0], [w, 0], [w, h], [0, h]];
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i], [u, v] = quad[i];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    const s = solve(A, b);
    if (!s) return null;
    const H = [[s[0], s[1], s[2]], [s[3], s[4], s[5]], [s[6], s[7], 1]];
    return { H, Hi: invert3(H), w, h };
  }

  function invert3(m) {
    const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) return null;
    return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
            [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
            [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
  }

  function apply(M, x, y) {
    const z = M[2][0] * x + M[2][1] * y + M[2][2];
    return [(M[0][0] * x + M[0][1] * y + M[0][2]) / z,
            (M[1][0] * x + M[1][1] * y + M[1][2]) / z];
  }

  function totalLanes() {
    return ed.layout.blocks.reduce(
      (n, b) => n + b.rows.reduce((m, r) => m + (r.lanes | 0), 0), 0);
  }

  /* ---------- server calls ---------- */

  async function propose(source) {
    const preset = $('lay-preset').value;
    try { localStorage.setItem('ssp.preset', preset); } catch (e) { /* private mode */ }
    const gen = ed.gen;
    setBusy(true, source === 'detect' ? 'detecting layout…' : 'loading layout…');
    try {
      const d = await ed.api(`/api/uploads/${ed.uid}/layout`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset, source }),
      });
      if (gen !== ed.gen) return;                 // editor closed or reopened
      ed.layout = d.layout;
      showWarnings(d.warnings, d.source === 'even'
        ? 'Even split of the whole photo: fit every outline and cut by hand.' : '');
      afterChange();
    } catch (e) {
      if (gen === ed.gen) showWarnings([e.message]);
    } finally { if (gen === ed.gen) setBusy(false); }
  }

  function schedulePreview() {
    clearTimeout(ed.previewTimer);
    ed.previewTimer = setTimeout(refreshPreview, 450);
  }

  async function refreshPreview() {
    if (!ed.uid) return;
    const gen = ed.gen;
    const seq = ++ed.previewSeq;
    const stale = () => gen !== ed.gen || seq !== ed.previewSeq;
    $('lay-preview-status').textContent = 'updating…';
    try {
      const res = await fetch(`/api/uploads/${ed.uid}/preview`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layout: ed.layout }),
      });
      if (stale()) return;            // a newer request, or another editor
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        if (!stale()) $('lay-preview-status').textContent = body.detail || res.statusText;
        return;
      }
      const blob = await res.blob();
      if (stale()) return;
      const url = URL.createObjectURL(blob);
      if (ed.previewUrl) URL.revokeObjectURL(ed.previewUrl);
      ed.previewUrl = url;
      $('lay-preview').src = url;
      $('lay-preview-status').textContent = '';
    } catch (e) {
      if (!stale()) $('lay-preview-status').textContent = e.message;
    }
  }

  /* ---------- UI state ---------- */

  function setBusy(on, msg = '') {
    for (const id of ['lay-detect', 'lay-last', 'lay-even', 'lay-confirm', 'lay-preset']) {
      $(id).disabled = on || (id === 'lay-last' && !hasLast());
    }
    $('lay-status').textContent = msg;
  }

  function hasLast() {
    return !!(ed.info && (ed.info.last_layouts || []).includes($('lay-preset').value));
  }

  function showWarnings(list, note = '') {
    const box = $('lay-warnings');
    box.innerHTML = '';
    for (const w of [note, ...(list || [])].filter(Boolean)) {
      const div = document.createElement('div');
      div.textContent = '⚠ ' + w;
      box.appendChild(div);
    }
  }

  function afterChange({ preview = true } = {}) {
    const n = totalLanes();
    const el = $('lay-count');
    el.textContent = `${n} lanes`;
    el.className = n === 96 ? 'ok' : 'bad';
    renderRowTable();
    draw();
    if (preview) schedulePreview();
  }

  function renderRowTable() {
    const tbody = $('lay-rows-body');
    tbody.innerHTML = '';
    let mix = 1;
    ed.layout.blocks.forEach((b, bi) => b.rows.forEach((r, ri) => {
      const tr = document.createElement('tr');
      const first = mix, last = mix + (r.lanes | 0) - 1;
      mix = last + 1;
      tr.innerHTML = `<td>piece ${bi + 1}</td><td>row ${ri + 1}</td>
        <td>mix ${first}–${last}</td>
        <td><input type="number" min="1" max="64" value="${r.lanes}"></td>`;
      tr.querySelector('input').addEventListener('change', (ev) => {
        const v = Math.max(1, Math.min(64, parseInt(ev.target.value, 10) || 1));
        r.lanes = v;
        afterChange();
      });
      tbody.appendChild(tr);
    }));
  }

  /* ---------- drawing ---------- */

  function fitScale() {          // whole photo visible: fit width and height
    const cont = $('layout-container');
    const maxH = window.innerHeight * 0.72 - 4;   // matches the CSS max-height
    return Math.max(0.05, Math.min((cont.clientWidth - 24) / ed.img.naturalWidth,
                                   maxH / ed.img.naturalHeight));
  }

  function handles() {
    const out = [];
    ed.layout.blocks.forEach((b, bi) => {
      const g = homography(b.quad);
      if (!g) return;
      b.quad.forEach((p, i) => out.push({ type: 'corner', b: bi, i, x: p[0], y: p[1] }));
      const fracs = [...new Set(b.rows.flatMap((r) => [r.top, r.bottom]))].sort((a, c) => a - c);
      for (const f of fracs) {
        const [x, y] = apply(g.H, g.w * 0.5, f * g.h);
        out.push({ type: 'cut', b: bi, frac: f, x, y });
      }
      b.rows.forEach((r, ri) => {
        const ym = ((r.top + r.bottom) / 2) * g.h;
        for (const side of ['left', 'right']) {
          const [x, y] = apply(g.H, r[side] * g.w, ym);
          out.push({ type: side, b: bi, r: ri, x, y });
        }
      });
    });
    return out;
  }

  function draw() {
    if (!ed.img || !ed.layout) return;
    const canvas = $('layout-canvas');
    const ctx = canvas.getContext('2d');
    const W = ed.img.naturalWidth, H = ed.img.naturalHeight;
    if (ed.scale === null) ed.scale = fitScale();
    const s = ed.scale;
    canvas.width = Math.round(W * s);
    canvas.height = Math.round(H * s);
    ctx.drawImage(ed.img, 0, 0, W * s, H * s);
    const P = (pt) => [pt[0] * s, pt[1] * s];
    let mix = 1;

    ed.layout.blocks.forEach((b, bi) => {
      const g = homography(b.quad);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ff3030';
      ctx.beginPath();
      b.quad.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath(); ctx.stroke();
      if (!g) return;
      b.rows.forEach((r, ri) => {
        const col = COLORS[(bi * b.rows.length + ri) % COLORS.length];
        const x0 = r.left * g.w, x1 = r.right * g.w, y0 = r.top * g.h, y1 = r.bottom * g.h;
        const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => P(apply(g.H, x, y)));
        ctx.strokeStyle = col; ctx.lineWidth = 1.5;
        ctx.beginPath();
        corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath(); ctx.stroke();
        const n = Math.max(1, r.lanes | 0), step = (x1 - x0) / n;
        const tick = (y1 - y0) * 0.1;
        const fontPx = Math.max(9, Math.min(16, step * s * 0.5));
        ctx.font = `bold ${fontPx}px sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.lineWidth = 1;
        for (let k = 0; k <= n; k++) {
          const [ax, ay] = P(apply(g.H, x0 + k * step, y0));
          const [bx, by] = P(apply(g.H, x0 + k * step, y0 + tick));
          ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        }
        for (let k = 0; k < n; k++) {
          const [cx, cy] = P(apply(g.H, x0 + (k + 0.5) * step, y0 + tick * 0.5));
          ctx.lineWidth = 3; ctx.strokeStyle = '#000';
          ctx.strokeText(String(mix + k), cx, cy);
          ctx.fillStyle = col;
          ctx.fillText(String(mix + k), cx, cy);
        }
        ctx.strokeStyle = col;
        mix += n;
      });
    });

    for (const h of handles()) {
      const [x, y] = P([h.x, h.y]);
      ctx.beginPath();
      if (h.type === 'corner') {
        ctx.arc(x, y, HANDLE_R, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255,48,48,0.85)';
      } else if (h.type === 'cut') {
        ctx.rect(x - 11, y - 4, 22, 8);
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
      } else {
        const d = h.type === 'left' ? -1 : 1;
        ctx.moveTo(x + d * 8, y); ctx.lineTo(x - d * 4, y - 7); ctx.lineTo(x - d * 4, y + 7);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
      }
      ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = '#000'; ctx.stroke();
    }
  }

  /* ---------- dragging ---------- */

  function toImage(ev) {
    const rect = $('layout-canvas').getBoundingClientRect();
    return [(ev.clientX - rect.left) / ed.scale, (ev.clientY - rect.top) / ed.scale];
  }

  function onDown(ev) {
    if (!ed.layout) return;
    const [ix, iy] = toImage(ev);
    const reach = (HANDLE_R + 4) / ed.scale;
    let best = null, bestD = Infinity;
    for (const h of handles()) {
      const d = Math.hypot(h.x - ix, h.y - iy);
      if (d < reach && d < bestD) { best = h; bestD = d; }
    }
    if (!best) return;
    ev.preventDefault();
    ed.drag = { ...best, rowOnly: ev.shiftKey };
    $('layout-canvas').setPointerCapture(ev.pointerId);
  }

  function onMove(ev) {
    const canvas = $('layout-canvas');
    if (!ed.drag) {           // hover cursor
      const [ix, iy] = toImage(ev);
      const reach = (HANDLE_R + 4) / ed.scale;
      const over = ed.layout && handles().some((h) => Math.hypot(h.x - ix, h.y - iy) < reach);
      canvas.style.cursor = over ? 'grab' : 'default';
      return;
    }
    const [ix, iy] = toImage(ev);
    const W = ed.img.naturalWidth, H = ed.img.naturalHeight;
    const d = ed.drag, b = ed.layout.blocks[d.b];
    if (d.type === 'corner') {
      const old = b.quad[d.i];
      b.quad[d.i] = [Math.max(0, Math.min(W, ix)), Math.max(0, Math.min(H, iy))];
      if (!homography(b.quad) || !convex(b.quad)) b.quad[d.i] = old;
    } else {
      const g = homography(b.quad);
      if (!g || !g.Hi) return;
      const [rx, ry] = apply(g.Hi, ix, iy);
      if (d.type === 'cut') {
        const fracs = [...new Set(b.rows.flatMap((r) => [r.top, r.bottom]))].sort((a, c) => a - c);
        const i = fracs.findIndex((f) => Math.abs(f - d.frac) < 1e-9);
        const lo = i > 0 ? fracs[i - 1] + MIN_GAP : 0;
        const hi = i < fracs.length - 1 ? fracs[i + 1] - MIN_GAP : 1;
        const nf = round4(Math.max(lo, Math.min(hi, ry / g.h)));
        for (const r of b.rows) {
          if (Math.abs(r.top - d.frac) < 1e-9) r.top = nf;
          if (Math.abs(r.bottom - d.frac) < 1e-9) r.bottom = nf;
        }
        d.frac = nf;
      } else {
        const rows = d.rowOnly ? [b.rows[d.r]] : b.rows;
        const v = rx / g.w;
        for (const r of rows) {
          if (d.type === 'left') r.left = round4(Math.max(0, Math.min(r.right - MIN_GAP, v)));
          else r.right = round4(Math.min(1, Math.max(r.left + MIN_GAP, v)));
        }
      }
    }
    draw();
  }

  function onUp() {
    if (!ed.drag) return;
    ed.drag = null;
    afterChange();
  }

  function convex(q) {             // clockwise convex in image coords (y down)
    for (let i = 0; i < 4; i++) {
      const [a, b, c] = [q[i], q[(i + 1) % 4], q[(i + 2) % 4]];
      if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0) return false;
    }
    return true;
  }

  const round4 = (v) => Math.round(v * 1e4) / 1e4;

  /* ---------- public ---------- */

  async function open(info, { layout = null, warnings = [], api, onConfirm, onCancel } = {}) {
    const gen = ++ed.gen;
    ed.uid = info.upload_id; ed.info = info; ed.api = api;
    ed.onConfirm = onConfirm; ed.onCancel = onCancel;
    ed.scale = null; ed.layout = null;
    $('placeholder').hidden = true;
    $('workspace').hidden = true;
    $('layout-step').hidden = false;
    $('lay-title').textContent = `Layout: ${info.image_name}`;
    $('lay-preview').removeAttribute('src');
    showWarnings([]);                        // nothing stale from a previous run
    $('lay-count').textContent = '';
    $('lay-count').className = '';
    $('lay-rows-body').innerHTML = '';
    $('layout-canvas').width = 0;
    ed.img = new Image();
    ed.img.src = `/api/uploads/${ed.uid}/image`;
    await ed.img.decode();
    if (gen !== ed.gen) return;               // closed or reopened meanwhile
    let preset = layout && layout.preset;
    if (!preset) { try { preset = localStorage.getItem('ssp.preset'); } catch (e) { /* */ } }
    $('lay-preset').value = (info.presets || []).includes(preset) ? preset : '3x32';
    setBusy(false);
    if (layout) {
      ed.layout = layout;
      showWarnings(warnings, 'Re-cutting a previous session: the result will be a new session.');
      afterChange();
    } else {
      await propose('detect');
    }
  }

  function close() {
    ed.gen++;                                 // pending work belongs to no one now
    $('layout-step').hidden = true;
    clearTimeout(ed.previewTimer);
    if (ed.previewUrl) URL.revokeObjectURL(ed.previewUrl);
    ed.previewUrl = null;
    ed.layout = null; ed.uid = null;
  }

  function bind() {
    $('lay-preset').addEventListener('change', () => propose('detect'));
    $('lay-detect').addEventListener('click', () => propose('detect'));
    $('lay-last').addEventListener('click', () => propose('last'));
    $('lay-even').addEventListener('click', () => propose('even'));
    $('lay-zoom-in').addEventListener('click', () => { ed.scale = Math.min(3, (ed.scale || fitScale()) * 1.25); draw(); });
    $('lay-zoom-out').addEventListener('click', () => { ed.scale = Math.max(0.05, (ed.scale || fitScale()) * 0.8); draw(); });
    $('lay-zoom-fit').addEventListener('click', () => { ed.scale = null; draw(); });
    window.addEventListener('resize', () => { if (ed.layout && ed.drag === null) draw(); });
    $('lay-cancel').addEventListener('click', () => { close(); if (ed.onCancel) ed.onCancel(); });
    $('lay-confirm').addEventListener('click', async () => {
      if (!ed.layout || !ed.onConfirm) return;
      // the session consumes the staged photo: no preview may follow
      clearTimeout(ed.previewTimer);
      ed.previewSeq++;
      const gen = ed.gen;
      setBusy(true, 'segmenting + typing… (a few seconds)');
      try {
        await ed.onConfirm(ed.uid, JSON.parse(JSON.stringify(ed.layout)));
        if (gen === ed.gen) close();
      } catch (e) {
        if (gen === ed.gen) showWarnings([e.message]);
      } finally { if (gen === ed.gen) setBusy(false); }
    });
    const c = $('layout-canvas');
    c.addEventListener('pointerdown', onDown);
    c.addEventListener('pointermove', onMove);
    c.addEventListener('pointerup', onUp);
    c.addEventListener('pointercancel', onUp);
  }

  // exposed for tests / debugging
  return { open, close, bind, _geom: { blockSize, homography, apply, convex },
           _state: () => ed };
})();
