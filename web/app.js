/* Data-association workbench front-end (no dependencies).
 *
 * Renders one frame at a time: synthetic detections, committed / provisional
 * tracks, the alternative hypotheses that are still in the window, and the
 * append-only ledger of sealing, replays and corrections.
 *
 * Colour encodes target identity; it is never the only encoding - every track
 * head carries its id label, every panel shows ids as text, and the detections
 * table is the accessible table view.
 */
"use strict";

const PALETTE_SLOTS = 3; // validated all-pairs categorical set (blue/orange/aqua)

const S = {
  catalog: [],
  scenario: "crossing",
  mode: "window",
  sessionId: null,
  overview: null,
  history: [],
  frameCache: new Map(),
  altCache: new Map(),
  frame: 0,
  playing: false,
  timer: null,
  epoch: 0,
  showTruth: false,
  tab: "branches",
  selectedDet: null,
  selectedTrack: null,
  selectedAlt: null,
  lockPrefill: null,
  colorOf: new Map(),
  paths: new Map(),
  hit: [],
  mounted: false,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function esc(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

/* ---------- API ---------- */
const API = {
  async request(path, options = {}) {
    const res = await fetch(path, options);
    let payload = null;
    try {
      payload = await res.json();
    } catch (_err) {
      payload = null;
    }
    if (!res.ok || (payload && payload.ok === false)) {
      const msg = (payload && payload.error) || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return payload;
  },
  get(path) {
    return this.request(path);
  },
  post(path, body) {
    return this.request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  },
  del(path, body) {
    return this.request(path, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  },
};

let toastTimer = null;
function toast(message, kind = "") {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show ${kind}`.trim();
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.className = "toast";
  }, 3600);
}

/* ---------- colour helpers ---------- */
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function paletteColors() {
  return [cssVar("--s1"), cssVar("--s2"), cssVar("--s3")];
}

function buildColorMap() {
  const colors = paletteColors();
  S.colorOf = new Map();
  S.paths = new Map();
  let slot = 0;
  const seen = new Set();
  for (const entry of S.history) {
    for (const tr of entry.tracks) {
      if (!tr.confirmed || seen.has(tr.id)) continue;
      seen.add(tr.id);
      S.colorOf.set(tr.id, slot < PALETTE_SLOTS ? colors[slot] : null);
      slot += 1;
    }
  }
  for (const entry of S.history) {
    for (const tr of entry.tracks) {
      let list = S.paths.get(tr.id);
      if (!list) {
        list = [];
        S.paths.set(tr.id, list);
      }
      list.push({ frame: entry.frame, x: tr.x, y: tr.y, confirmed: tr.confirmed, det: tr.det });
    }
  }
}

function trackColor(tid, confirmed = true) {
  if (!confirmed) return cssVar("--muted");
  const c = S.colorOf.get(tid);
  return c === undefined || c === null ? cssVar("--muted") : c;
}

/* ---------- session lifecycle ---------- */
async function boot() {
  bindChrome();
  try {
    const data = await API.get("/api/scenarios");
    S.catalog = data.catalog || [];
  } catch (err) {
    toast(`无法读取场景列表：${err.message}`, "crit");
    return;
  }
  const select = $("#scenario-select");
  select.innerHTML = S.catalog
    .map((s) => `<option value="${esc(s.key)}">${esc(s.name)}</option>`)
    .join("");
  select.value = S.scenario;
  await createSession();
}

async function createSession() {
  try {
    const data = await API.post("/api/sessions", { scenario: S.scenario, mode: S.mode });
    S.sessionId = data.overview.session_id;
    S.frame = 0;
    S.frameCache.clear();
    S.altCache.clear();
    S.selectedDet = null;
    S.selectedTrack = null;
    S.selectedAlt = null;
    await absorb(data.overview, true);
  } catch (err) {
    toast(`创建会话失败：${err.message}`, "crit");
  }
}

async function absorb(overview, resetFrame = false) {
  S.overview = overview;
  S.view = null;
  S.epoch += 1;
  const n = overview.scenario.n_frames;
  if (resetFrame) S.frame = 0;
  S.frame = Math.max(0, Math.min(S.frame, n - 1));
  $("#frame-max").textContent = String(n - 1);
  $("#scrub").max = String(n - 1);
  $("#canvas-empty").classList.add("hide");
  await refreshHistory();
  buildTimeline();
  renderChrome();
  await renderCurrentFrame();
}

async function refreshHistory() {
  if (!S.sessionId) return;
  try {
    const data = await API.get(`/api/sessions/${S.sessionId}/tracks`);
    S.history = data.history || [];
    buildColorMap();
  } catch (err) {
    S.history = [];
    toast(`轨迹历史读取失败：${err.message}`, "crit");
  }
}

async function fetchFrame(frame) {
  if (S.frameCache.has(frame)) return S.frameCache.get(frame);
  const data = await API.get(`/api/sessions/${S.sessionId}/frame?frame=${frame}`);
  S.frameCache.set(frame, data.view);
  return data.view;
}

/* ---------- chrome ---------- */
function bindChrome() {
  $("#scenario-select").addEventListener("change", async (ev) => {
    S.scenario = ev.target.value;
    await createSession();
  });

  for (const mode of ["window", "greedy"]) {
    $(`#mode-${mode}`).addEventListener("click", async () => {
      if (S.mode === mode || !S.sessionId) return;
      S.mode = mode;
      try {
        // Rebuild the same session so existing locks keep applying.
        const data = await API.post(`/api/sessions/${S.sessionId}/rebuild`, { mode });
        S.frameCache.clear();
        S.altCache.clear();
        await absorb(data.overview);
        toast(mode === "window" ? "已切换到窗口推断（锁定保留）" : "已切换到局部贪心（锁定保留）");
      } catch (err) {
        S.mode = mode === "window" ? "greedy" : "window";
        toast(`切换模式失败：${err.message}`, "crit");
      }
    });
  }

  $("#compare-btn").addEventListener("click", openCompare);
  $("#compare-close").addEventListener("click", () => $("#compare-dialog").close());
  $("#export-btn").addEventListener("click", exportJson);

  $("#first-btn").addEventListener("click", () => go(0));
  $("#prev-btn").addEventListener("click", () => go(S.frame - 1));
  $("#next-btn").addEventListener("click", () => go(S.frame + 1));
  $("#last-btn").addEventListener("click", () => go(maxFrame()));
  $("#play-btn").addEventListener("click", togglePlay);

  $("#scrub").addEventListener("input", (ev) => go(parseInt(ev.target.value, 10)));

  for (const tab of $$(".tab")) {
    tab.addEventListener("click", () => {
      S.tab = tab.dataset.tab;
      for (const other of $$(".tab")) other.classList.toggle("on", other === tab);
      for (const panel of $$(".panel")) {
        panel.classList.toggle("on", panel.id === `panel-${S.tab}`);
      }
      renderPanels();
    });
  }

  const canvas = $("#canvas");
  canvas.addEventListener("mousemove", onCanvasMove);
  canvas.addEventListener("mouseleave", hideTip);
  canvas.addEventListener("click", onCanvasClick);

  window.addEventListener("resize", debounce(renderCanvas, 120));
  document.addEventListener("keydown", (ev) => {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      go(S.frame - 1);
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      go(S.frame + 1);
    } else if (ev.key === " ") {
      ev.preventDefault();
      togglePlay();
    }
  });
}

function debounce(fn, ms) {
  let handle = null;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), ms);
  };
}

function maxFrame() {
  return S.overview ? S.overview.scenario.n_frames - 1 : 0;
}

async function go(frame) {
  if (!S.sessionId) return;
  const clamped = Math.max(0, Math.min(frame, maxFrame()));
  S.frame = clamped;
  $("#scrub").value = String(clamped);
  await renderCurrentFrame();
}

function togglePlay() {
  const btn = $("#play-btn");
  if (S.playing) {
    clearInterval(S.timer);
    S.playing = false;
    btn.textContent = "播放";
    btn.setAttribute("aria-pressed", "false");
    return;
  }
  if (S.frame >= maxFrame()) S.frame = 0;
  S.playing = true;
  btn.textContent = "暂停";
  btn.setAttribute("aria-pressed", "true");
  S.timer = setInterval(async () => {
    if (S.frame >= maxFrame()) {
      togglePlay();
      return;
    }
    await go(S.frame + 1);
  }, 420);
}

function renderChrome() {
  const ov = S.overview;
  if (!ov) return;
  for (const mode of ["window", "greedy"]) {
    $(`#mode-${mode}`).classList.toggle("on", S.mode === mode);
  }
  $("#frame-num").textContent = String(S.frame);
  $("#foot-note").textContent = ov.truncated_search
    ? "窗口推断触发了搜索预算截断：被丢弃的假设记录在“封存与更正”里，结果不是全局最优。"
    : "窗口推断为有限预算搜索，不声称全局最优；封存后只追加更正关系。";
  renderStatusChips();
  const legend = $("#legend");
  const tracks = [...S.colorOf.entries()].slice(0, PALETTE_SLOTS);
  const hasOther = [...S.colorOf.values()].some((c) => c === null);
  const items = [
    ...tracks.map(([tid, color]) =>
      `<span class="item"><span class="swatch" style="background:${color}"></span>${esc(tid)}</span>`),
    hasOther
      ? `<span class="item"><span class="swatch line dashed"></span>其他轨迹（虚线 + 标签）</span>`
      : "",
    `<span class="item"><span class="swatch dot"></span>检测点</span>`,
    `<span class="item"><span class="swatch ring"></span>杂点 / 未关联</span>`,
    `<span class="item"><span class="swatch lock"></span>人工锁定</span>`,
    `<label class="item truth-toggle"><input id="truth-toggle" type="checkbox" ${S.showTruth ? "checked" : ""}> 显示真值</label>`,
  ].filter(Boolean);
  legend.innerHTML = items.join("");
  const toggle = $("#truth-toggle");
  if (toggle) {
    toggle.addEventListener("change", (ev) => {
      S.showTruth = ev.target.checked;
      renderCanvas();
      if (S.tab === "detections") renderPanels();
    });
  }
}

function renderStatusChips() {
  const ov = S.overview;
  if (!ov) return;
  const entry = (S.view && S.view.tracker) || {};
  const alts = ((S.view && S.view.tracker && S.view.tracker.alternatives) || []).length;
  const chips = [
    `<span class="chip strong">${S.mode === "window" ? "窗口推断" : "局部贪心"}</span>`,
    `<span class="chip">封存到第 ${ov.sealed_until} 帧</span>`,
    entry.sealed
      ? `<span class="chip ok">本帧已封存</span>`
      : entry.provisional
      ? `<span class="chip warn">本帧在窗口内（暂定）</span>`
      : "",
    alts > 1 ? `<span class="chip">${alts} 个假设</span>` : "",
    entry.late ? `<span class="chip warn">迟到帧</span>` : "",
    entry.replayed ? `<span class="chip">已回溯重放</span>` : "",
    entry.sealed_kind === "late_correction" ? `<span class="chip warn">迟到·仅追加更正</span>` : "",
    (S.view && S.view.pending) ? `<span class="chip warn">按计划挂起</span>` : "",
    (ov.locks || []).length ? `<span class="chip">锁定 ${ov.locks.length}</span>` : "",
    (ov.conflicts || []).length ? `<span class="chip crit">冲突 ${ov.conflicts.length}</span>` : "",
    ov.truncated_search ? `<span class="chip crit">搜索已截断 · 非全局最优</span>` : "",
  ].filter(Boolean);
  $("#status-chips").innerHTML = chips.join("");
  const scenario = ov.scenario || {};
  $("#stage-hint").textContent = `第 ${S.frame} 帧 · ${scenario.notes || scenario.description || ""}`;
}

/* ---------- current frame ---------- */
let renderToken = 0;

async function renderCurrentFrame() {
  if (!S.sessionId) return;
  const token = ++renderToken;
  let view = null;
  try {
    view = await fetchFrame(S.frame);
  } catch (err) {
    toast(`读取第 ${S.frame} 帧失败：${err.message}`, "crit");
    return;
  }
  if (token !== renderToken) return;
  S.view = view;
  S.selectedAlt = null;
  S.altDetail = null;
  S.selectedTrack = null;
  renderChrome();
  renderTimelineCurrent();
  renderCanvas();
  renderPanels();
}

function currentNode(view) {
  if (!view || !view.tracker) return null;
  return view.tracker.committed || (view.tracker.alternatives || [])[0] || null;
}

function nodeTracks(view) {
  const node = currentNode(view);
  return (node && node.tracks) || [];
}

/* ---------- canvas ---------- */
function canvasMetrics() {
  const canvas = $("#canvas");
  const rect = canvas.getBoundingClientRect();
  const w = rect.width || 700;
  const h = rect.height || w;
  const pad = Math.max(16, Math.min(w, h) * 0.05);
  const span = Math.min(w, h) - pad * 2;
  return { canvas, rect, w, h, pad, scale: span / 100 };
}

function project(x, y, m) {
  return {
    x: m.pad + x * m.scale,
    y: m.h - m.pad - y * m.scale,
  };
}

function renderCanvas() {
  const m = canvasMetrics();
  const { canvas, w, h } = m;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const surface = "#141413";
  const ink = cssVar("--ink");
  const ink2 = cssVar("--ink-2");
  const muted = cssVar("--muted");
  const axis = cssVar("--axis");
  const warning = cssVar("--warning");

  // Grid (recessive).
  ctx.lineWidth = 1;
  ctx.strokeStyle = axis;
  for (let v = 0; v <= 100; v += 20) {
    const a = project(v, 0, m);
    const b = project(v, 100, m);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    const c = project(0, v, m);
    const d = project(100, v, m);
    ctx.beginPath();
    ctx.moveTo(c.x, c.y);
    ctx.lineTo(d.x, d.y);
    ctx.stroke();
  }
  ctx.fillStyle = muted;
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("0", m.pad - 2, h - m.pad + 12);
  ctx.fillText("100", w - m.pad - 16, h - m.pad + 12);

  const view = S.view;
  if (!view) return;

  // Truth (evaluation aid only).
  if (S.showTruth) {
    ctx.strokeStyle = muted;
    ctx.lineWidth = 1.2;
    for (const t of view.truth || []) {
      const p = project(t.x, t.y, m);
      ctx.strokeRect(p.x - 3.5, p.y - 3.5, 7, 7);
      ctx.fillStyle = muted;
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(t.truth_id, p.x + 6, p.y - 5);
    }
  }

  const node = currentNode(view);
  const liveIds = new Set(nodeTracks(view).map((t) => t.id));

  // Track tails from history (confirmed = full, tentative = short stub).
  const tentativeStubs = new Set();
  if (node) {
    for (const tr of node.tracks) {
      if (!tr.confirmed && tr.det) {
        tentativeStubs.add(tr.id);
        const list = S.paths.get(tr.id) || [];
        const tail = list.slice(-3).map((p) => project(p.x, p.y, m));
        if (tail.length > 1) {
          ctx.save();
          ctx.globalAlpha = 0.28;
          ctx.strokeStyle = muted;
          ctx.lineWidth = 1.2;
          ctx.setLineDash([2, 3]);
          ctx.beginPath();
          tail.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
          ctx.stroke();
          ctx.restore();
        }
      }
    }
  }

  const confirmedTracks = [...S.colorOf.keys()];
  for (const tid of confirmedTracks) {
    const list = (S.paths.get(tid) || []).filter((p) => p.frame <= S.frame);
    if (list.length < 2) continue;
    const color = trackColor(tid, true);
    const other = S.colorOf.get(tid) === null;
    const segments = [];
    let current = [];
    let prevFrame = null;
    for (const p of list) {
      if (prevFrame != null && p.frame - prevFrame > 1) {
        if (current.length) segments.push(current);
        current = [];
      }
      current.push(project(p.x, p.y, m));
      prevFrame = p.frame;
    }
    if (current.length) segments.push(current);
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (const seg of segments) {
      // 2px surface halo keeps crossing lines separable.
      ctx.strokeStyle = surface;
      ctx.lineWidth = 5;
      ctx.beginPath();
      seg.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.92;
      if (other) ctx.setLineDash([5, 3]);
      ctx.beginPath();
      seg.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  }

  // Detection hit targets.
  S.hit = [];
  const assignment = (node && node.assignment) || {};
  const trackState = new Map(nodeTracks(view).map((t) => [t.id, t]));
  const lockedDets = new Set((view.locks || []).map((l) => l.det_id));

  for (const det of view.detections || []) {
    const p = project(det.x, det.y, m);
    const tid = assignment[det.id];
    const tr = tid ? trackState.get(tid) : null;
    const isLocked = lockedDets.has(det.id);
    S.hit.push({ id: det.id, x: p.x, y: p.y, det });

    if (tid) {
      const color = trackColor(tid, tr ? tr.confirmed : true);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = surface;
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = surface;
      ctx.fill();
      ctx.lineWidth = 1.3;
      ctx.strokeStyle = muted;
      ctx.stroke();
    }

    if (isLocked) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = warning;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (S.selectedDet === det.id) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10.5, 0, Math.PI * 2);
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = ink;
      ctx.stroke();
    }
  }

  // Track heads + direct labels at the current frame.
  if (node) {
    for (const tr of node.tracks) {
      if (!tr.confirmed && !tentativeStubs.has(tr.id) && !liveIds.has(tr.id)) continue;
      const p = project(tr.x, tr.y, m);
      const color = trackColor(tr.id, tr.confirmed);
      ctx.save();
      ctx.globalAlpha = tr.confirmed ? 1 : 0.45;
      ctx.beginPath();
      ctx.arc(p.x, p.y, tr.confirmed ? 4.6 : 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = surface;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillStyle = ink2;
      const label = tr.id;
      ctx.fillText(label, p.x + 7, p.y - 6);
      if (S.selectedTrack === tr.id) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 11, 0, Math.PI * 2);
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  if (S.selectedDet) {
    const hit = S.hit.find((h) => h.id === S.selectedDet);
    if (hit) {
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillStyle = ink;
      ctx.fillText(hit.id, hit.x + 12, hit.y + 16);
    }
  }
}

function tipNode() {
  let node = $("#chart-tip");
  if (!node) {
    node = document.createElement("div");
    node.id = "chart-tip";
    node.className = "chart-tip";
    $(".canvas-wrap").appendChild(node);
  }
  return node;
}

function hideTip() {
  const tip = $("#chart-tip");
  if (tip) tip.classList.remove("show");
}

function onCanvasMove(ev) {
  const canvas = $("#canvas");
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  let best = null;
  let bestD = 13;
  for (const hit of S.hit) {
    const d = Math.hypot(hit.x - x, hit.y - y);
    if (d < bestD) {
      bestD = d;
      best = hit;
    }
  }
  if (!best) {
    hideTip();
    return;
  }
  const tip = tipNode();
  const view = S.view || {};
  const node = currentNode(view);
  const assignment = (node && node.assignment) || {};
  const tid = assignment[best.id];
  const audit = ((view.tracker && view.tracker.audit) || []).find(
    (a) => a.det === best.id && a.chosen
  );
  const truth = S.showTruth ? (best.det.truth_id || "杂点") : null;
  const rows = [
    `<div class="mono">${esc(best.id)}</div>`,
    `<div><span class="k">坐标</span> (${fmt(best.det.x)}, ${fmt(best.det.y)})</div>`,
    truth != null ? `<div><span class="k">真值</span> ${esc(truth)}</div>` : "",
    `<div><span class="k">关联</span> ${tid ? esc(tid) : "未关联（杂点 / 待新建）"}</div>`,
    audit
      ? `<div class="why">Δ²=${fmt(audit.d2, 2)} · 代价 ${fmt(audit.cost, 2)}</div>`
      : "",
  ].filter(Boolean);
  tip.innerHTML = rows.join("");
  const wrap = $(".canvas-wrap").getBoundingClientRect();
  const px = Math.min(Math.max(best.x + 14, 6), wrap.width - 200);
  const py = Math.max(best.y - 10, 6);
  tip.style.left = `${px}px`;
  tip.style.top = `${py}px`;
  tip.classList.add("show");
}

function onCanvasClick(ev) {
  const canvas = $("#canvas");
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  let best = null;
  let bestD = 13;
  for (const hit of S.hit) {
    const d = Math.hypot(hit.x - x, hit.y - y);
    if (d < bestD) {
      bestD = d;
      best = hit;
    }
  }
  if (!best) return;
  S.selectedDet = S.selectedDet === best.id ? null : best.id;
  selectTab("detections");
  renderCanvas();
  renderPanels();
}

function selectTab(name) {
  S.tab = name;
  for (const tab of $$(".tab")) tab.classList.toggle("on", tab.dataset.tab === name);
  for (const panel of $$(".panel")) panel.classList.toggle("on", panel.id === `panel-${name}`);
}

/* ---------- timeline ---------- */
function buildTimeline() {
  const box = $("#timeline");
  if (!S.overview) return;
  box.innerHTML = S.overview.frames
    .map((f) => {
      const cls = ["seg"];
      if (f.sealed_kind === "late_correction") cls.push("sealed", "late");
      else if (f.late) cls.push("late");
      if (f.replayed) cls.push("replay");
      if ((f.alternatives || 0) > 1) cls.push("branch");
      else if (f.sealed) cls.push("sealed");
      else if (f.provisional) cls.push("open");
      const stateText = f.sealed_kind === "late_correction"
        ? "迟到·仅追加更正"
        : f.sealed
        ? "已封存"
        : "窗口开放";
      const tags = [
        stateText,
        f.replayed ? "回溯重放" : "",
        (f.alternatives || 0) > 1 ? `${f.alternatives} 个假设` : "",
      ]
        .filter(Boolean)
        .join(" · ");
      return `<span class="${cls.join(" ")}" data-frame="${f.frame}" title="第 ${f.frame} 帧：${esc(tags)}"></span>`;
    })
    .join("");
  box.onclick = (ev) => {
    const seg = ev.target.closest(".seg");
    if (seg) go(parseInt(seg.dataset.frame, 10));
  };
  renderTimelineCurrent();
}

function renderTimelineCurrent() {
  for (const seg of $$("#timeline .seg")) {
    seg.classList.toggle("current", parseInt(seg.dataset.frame, 10) === S.frame);
  }
}

/* ---------- panels ---------- */
function renderPanels() {
  if (!S.view) return;
  const map = {
    branches: renderBranches,
    detections: renderDetections,
    locks: renderLocks,
    ledger: renderLedger,
    metrics: renderMetrics,
  };
  (map[S.tab] || renderBranches)($(`#panel-${S.tab}`));
}

function alternatives(view) {
  return (view.tracker && view.tracker.alternatives) || [];
}

function renderBranches(panel) {
  const view = S.view;
  const ov = S.overview;
  const cfg = ov.config;
  const alts = alternatives(view);
  const entry = view.tracker || {};
  const parts = [];
  parts.push(
    `<p class="panel-note">窗口 ${cfg.window_len} 帧 · beam ${cfg.beam_width} · 每帧枚举 ${cfg.k_best} 个指派 · 分支阈值 ${fmt(cfg.branch_margin)} · 保留阈值 ${fmt(cfg.prune_margin)}</p>`
  );
  if (ov.truncated_search) {
    parts.push(
      `<p class="panel-note warn">搜索曾触及节点预算被截断：以下评分是“已评估假设中的最好者”，不是全局最优。</p>`
    );
  }
  if (!entry || entry === null) {
    parts.push(`<p class="empty">该帧尚未进入追踪器。</p>`);
    panel.innerHTML = parts.join("");
    return;
  }
  const state = entry.sealed ? "已封存" : entry.sealed_kind === "late_correction" ? "迟到·仅追加更正" : "窗口内（暂定）";
  parts.push(
    `<div class="section-head"><h3 style="margin-top:0">第 ${view.frame} 帧 · ${state}</h3><span class="count">${alts.length} 个假设</span></div>`
  );
  if (entry.sealed_kind === "late_correction") {
    parts.push(
      `<p class="panel-note warn">该帧迟到送达时，封存线已经越过它。它没有进入窗口，已封存的关联保持原样；“封存与更正”里的更正记录是唯一新增的信息。</p>`
    );
    panel.innerHTML = parts.join("");
    return;
  }
  if (!alts.length) {
    parts.push(`<p class="empty">没有存活的假设。</p>`);
  }
  const mapNode = entry.committed || alts[0] || null;
  alts.forEach((alt, idx) => {
    const isMap = alt.is_map;
    const outcome = alt.outcome || (isMap ? "committed" : "open");
    const diff = mapNode ? diffAssignments(mapNode.assignment, alt.assignment) : [];
    const outcomeLabel = {
      committed: "已提交",
      pruned: "已裁剪",
      open: "待比较",
    }[outcome] || outcome;
    const cls = ["alt-card"];
    if (isMap) cls.push("map");
    if (outcome === "pruned") cls.push("pruned");
    const gap = mapNode ? alt.score - mapNode.score : 0;
    parts.push(`
      <div class="${cls.join(" ")}" data-alt="${esc(alt.key)}">
        <div class="alt-head">
          <span class="alt-key">${esc(alt.key)}</span>
          <span class="alt-score">代价 ${fmt(alt.score, 2)}</span>
        </div>
        <div class="alt-meta">
          <span class="tag ${outcome === "committed" ? "committed" : outcome === "pruned" ? "pruned" : "open"}">${esc(outcomeLabel)}</span>
          ${isMap ? `<span class="tag map">MAP</span>` : `<span class="tag">与 MAP 差 ${fmt(gap, 2)}</span>`}
          <span>权重 ${alt.weight == null ? "—" : fmt(alt.weight * 100, 1) + "%"}</span>
        </div>
        <div class="weight-bar"><i style="width:${Math.max(1, (alt.weight || 0) * 100)}%"></i></div>
        <div class="alt-meta">
          <span>新建 ${(alt.births || []).length}</span>
          <span>结束 ${(alt.deaths || []).length}</span>
          ${(alt.deaths || []).length ? `<span class="mono">${esc((alt.deaths || []).join(", "))}</span>` : ""}
        </div>
        ${alt.outcome_reason ? `<div class="small" style="margin-top:5px">${esc(alt.outcome_reason)}</div>` : ""}
        ${
          diff.length
            ? `<div class="alt-detail"><div>与 MAP 的指派差异：</div><ol>${diff
                .map((d) => `<li><span class="mono">${esc(d.det)}</span>：${esc(d.a || "—")} → ${esc(d.b || "—")}</li>`)
                .join("")}</ol></div>`
            : `<div class="alt-detail muted">与 MAP 的指派相同（差异来自更早帧的轨迹状态）</div>`
        }
        <div class="btn-row">
          <button class="btn-xs" data-detail="${esc(alt.key)}">查看由来</button>
        </div>
        <div class="alt-detail-slot" id="alt-slot-${esc(alt.key)}"></div>
      </div>
    `);
  });
  parts.push(
    `<p class="panel-note">裁剪规则：分支只在帧代价与当前最优相差 ${fmt(cfg.branch_margin)} 内产生；累计代价落后超过 ${fmt(cfg.prune_margin)} 或超 beam 宽度的假设被丢弃，并在“封存与更正”里留档。</p>`
  );
  panel.innerHTML = parts.join("");
  for (const btn of $$("[data-detail]", panel)) {
    btn.addEventListener("click", () => showAltDetail(btn.dataset.detail));
  }
  if (S.altDetail && S.altDetail.key) renderAltDetail(S.altDetail);
}

function diffAssignments(a, b) {
  const out = [];
  const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
  for (const det of [...keys].sort()) {
    const ta = (a || {})[det] || null;
    const tb = (b || {})[det] || null;
    if (ta !== tb) out.push({ det, a: ta, b: tb });
  }
  return out;
}

async function showAltDetail(key) {
  const view = S.view;
  if (!view) return;
  const cacheKey = `${S.sessionId}:${view.frame}:${key}:${S.epoch}`;
  let detail = S.altCache.get(cacheKey);
  if (!detail) {
    try {
      const data = await API.get(
        `/api/sessions/${S.sessionId}/alternative?frame=${view.frame}&key=${encodeURIComponent(key)}`
      );
      detail = data.detail;
      if (detail) S.altCache.set(cacheKey, detail);
    } catch (err) {
      toast(`读取假设失败：${err.message}`, "crit");
      return;
    }
  }
  if (!detail) {
    toast("该假设没有留存的细节（可能只是舍入误差内的重复指派）");
    return;
  }
  S.altDetail = detail;
  renderAltDetail(detail);
}

function renderAltDetail(detail) {
  const slot = $(`#alt-slot-${CSS.escape(detail.key)}`);
  if (!slot) return;
  const hist = (detail.history || []).slice(-6);
  const rows = [];
  rows.push(`<div style="margin-top:4px">结果：<strong>${esc(detail.outcome || "—")}</strong> · ${esc(detail.outcome_reason || "")}</div>`);
  rows.push(`<div class="hist">子假设：${(detail.child_keys || []).length ? esc(detail.child_keys.join(", ")) : "无（该帧之后未被继续扩展）"}</div>`);
  rows.push(`<div style="margin-top:6px">最近 ${hist.length} 帧的指派：</div>`);
  rows.push(
    `<ol class="hist">${hist
      .map((h) => {
        const pairs = Object.entries(h.assignment || {})
          .map(([det, tid]) => `${det}→${tid}`)
          .join("  ");
        return `<li>第 ${h.frame} 帧 · 代价 ${fmt(h.score, 2)} · ${esc(pairs || "无关联")}</li>`;
      })
      .join("")}</ol>`
  );
  slot.innerHTML = `<div class="alt-detail">${rows.join("")}</div>`;
}

/* ---------- detections panel ---------- */
function renderDetections(panel) {
  const view = S.view;
  const node = currentNode(view);
  const entry = view.tracker || {};
  const assignment = (node && node.assignment) || {};
  const audit = entry.audit || [];
  const rejected = entry.rejected || [];
  const lockedDets = new Map((view.locks || []).map((l) => [l.det_id, l]));
  const parts = [];
  parts.push(
    `<p class="panel-note">${S.showTruth ? "真值列来自场景生成器，只用于核对；追踪器本身看不到它。" : "真值默认隐藏，可在画布图例中打开。"}</p>`
  );
  if (entry.sealed_kind === "late_correction") {
    parts.push(
      `<p class="panel-note warn">该帧迟到于封存线之后：它没有进入窗口，下面的“未关联”不代表杂点，只是没有参与关联。更正记录见“封存与更正”。</p>`
    );
  }

  const selected = S.selectedDet;
  const detRows = (view.detections || []).map((det) => {
    const tid = assignment[det.id];
    const lock = lockedDets.get(det.id);
    const isSelected = selected === det.id;
    const isClutter = !tid;
    return `<tr class="${isSelected ? "det-selected " : ""}${isClutter ? "det-clutter" : ""}" data-det="${esc(det.id)}">
      <td class="mono">${esc(det.id)}</td>
      <td class="mono">${fmt(det.x)}, ${fmt(det.y)}</td>
      ${S.showTruth ? `<td>${det.truth_id ? esc(det.truth_id) : "—"}</td>` : ""}
      <td>${tid ? `<span class="mono">${esc(tid)}</span>` : `<span class="muted">未关联</span>`}</td>
      <td>${lock ? `<span class="tag">锁 ${esc(lock.id)}</span>` : ""}${
        det.late ? `<span class="tag open">迟到</span>` : ""
      }</td>
      <td><button class="btn-xs" data-lock="${esc(det.id)}">锁定…</button></td>
    </tr>`;
  });

  parts.push(`<div class="section-head"><h3 style="margin-top:0">第 ${view.frame} 帧检测点</h3><span class="count">${(view.detections || []).length} 个</span></div>`);
  parts.push(`<table><thead><tr><th>检测点</th><th>坐标</th>${S.showTruth ? "<th>真值</th>" : ""}<th>关联轨迹</th><th></th><th></th></tr></thead><tbody>${detRows.join("") || `<tr><td colspan="6" class="empty">本帧没有检测点</td></tr>`}</tbody></table>`);

  if (selected) {
    const rows = audit.filter((a) => a.det === selected);
    parts.push(`<h3>为什么 ${esc(selected)} 这样处理</h3>`);
    if (!rows.length) {
      parts.push(`<p class="empty">该检测点没有进入任何配对（可能是杂点，或该帧尚未进入窗口）。</p>`);
    } else {
      parts.push(`<table><thead><tr><th>候选轨迹</th><th>Δ²</th><th>距离</th><th>代价</th><th>结论</th></tr></thead><tbody>
        ${rows
          .sort((a, b) => a.cost - b.cost)
          .map(
            (r) => `<tr class="${r.chosen ? "pair-chosen" : ""}${r.feasible ? "" : " pair-gated"}">
              <td class="mono">${esc(r.track)}</td>
              <td class="mono">${fmt(r.d2, 2)}</td>
              <td class="mono">${fmt(r.euclid, 2)}</td>
              <td class="mono">${fmt(r.cost, 2)}</td>
              <td>${esc(r.reason)}${r.forced ? ` <span class="tag">人工</span>` : ""}</td>
            </tr>`
          )
          .join("")}</tbody></table>`);
    }
    const rej = rejected.filter((r) => r.det === selected);
    if (rej.length) {
      parts.push(`<h3>贪心拒绝记录</h3>`);
      parts.push(rej.map((r) => `<div class="finding warn">${esc(r.detail)}</div>`).join(""));
    }
    parts.push(
      `<div class="btn-row"><button class="btn-xs" id="det-lock-btn">锁定 ${esc(selected)} 到某条轨迹</button><button class="btn-xs" id="det-clear">取消选中</button></div>`
    );
  }

  parts.push(`<h3>全部候选配对（含被排除原因）</h3>`);
  if (!audit.length) {
    parts.push(`<p class="empty">该帧没有可比较的（轨迹, 检测点）配对。</p>`);
  } else {
    parts.push(`<table><thead><tr><th>检测点</th><th>轨迹</th><th>Δ²</th><th>代价</th><th>结论</th></tr></thead><tbody>
      ${audit
        .map(
          (r) => `<tr class="${r.chosen ? "pair-chosen" : ""}${r.feasible ? "" : " pair-gated"}" data-pair-det="${esc(r.det)}" data-pair-track="${esc(r.track)}">
            <td class="mono">${esc(r.det)}</td>
            <td class="mono">${esc(r.track)}</td>
            <td class="mono">${fmt(r.d2, 2)}</td>
            <td class="mono">${fmt(r.cost, 2)}</td>
            <td>${esc(r.reason)}</td>
          </tr>`
        )
        .join("")}</tbody></table>`);
  }

  if (rejected.length) {
    parts.push(`<h3>贪心模式的逐帧拒绝记录</h3>`);
    parts.push(
      rejected
        .map((r) => `<div class="finding warn"><span class="mono">${esc(r.det)}</span> · ${esc(r.detail)}</div>`)
        .join("")
    );
  }

  panel.innerHTML = parts.join("");

  for (const row of $$("[data-det]", panel)) {
    row.addEventListener("click", (ev) => {
      if (ev.target.closest("button")) return;
      S.selectedDet = S.selectedDet === row.dataset.det ? null : row.dataset.det;
      renderCanvas();
      renderDetections(panel);
    });
  }
  for (const btn of $$("[data-lock]", panel)) {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      openLockForm(view.frame, btn.dataset.lock);
    });
  }
  for (const row of $$("[data-pair-det]", panel)) {
    row.addEventListener("click", () => {
      S.selectedDet = row.dataset.pairDet;
      renderCanvas();
      renderDetections(panel);
    });
  }
  const lockBtn = $("#det-lock-btn", panel);
  if (lockBtn) lockBtn.addEventListener("click", () => openLockForm(view.frame, selected));
  const clearBtn = $("#det-clear", panel);
  if (clearBtn)
    clearBtn.addEventListener("click", () => {
      S.selectedDet = null;
      renderCanvas();
      renderDetections(panel);
    });
}

function openLockForm(frame, detId) {
  S.lockPrefill = { frame, detId };
  selectTab("locks");
  renderPanels();
  const form = $("#lock-det");
  if (form) form.focus();
}

/* ---------- locks panel ---------- */
function renderLocks(panel) {
  const view = S.view;
  const ov = S.overview;
  const node = currentNode(view);
  const tracks = nodeTracks(view).map((t) => t.id);
  const extraTracks = new Set(tracks);
  for (const a of alternatives(view)) {
    for (const t of a.tracks || []) extraTracks.add(t.id);
  }
  const dets = (view.detections || []).map((d) => d.id);
  const prefill = S.lockPrefill || {};
  const frame = prefill.frame != null ? prefill.frame : view.frame;
  const detValue = prefill.detId || (S.selectedDet && S.frame === frame ? S.selectedDet : dets[0] || "");

  const parts = [];
  parts.push(
    `<p class="panel-note">锁定是“人工证据”，会作为强先验进入该帧及之后所有假设。锁定不保证一定成立：当检测点或轨迹不存在时，追踪器会明确指出是哪个检测点。</p>`
  );

  parts.push(`<h3 style="margin-top:0">新增锁定</h3>`);
  parts.push(`
    <div class="lock-form">
      <label class="small">帧 <input id="lock-frame" type="number" min="0" max="${maxFrame()}" value="${frame}" style="width:64px"></label>
      <label class="small">检测点
        <select id="lock-det">${dets.map((d) => `<option value="${esc(d)}" ${d === detValue ? "selected" : ""}>${esc(d)}</option>`).join("")}</select>
      </label>
      <label class="small">轨迹
        <select id="lock-track">${[...extraTracks].sort().map((t) => `<option value="${esc(t)}">${esc(t)}</option>`).join("")}</select>
      </label>
      <input id="lock-note" type="text" placeholder="备注（可选）">
      <button class="primary" id="lock-add" type="button">添加锁定</button>
    </div>
    <p class="small" id="lock-hint"></p>
  `);

  const locks = ov.locks || [];
  parts.push(`<div class="section-head"><h3>当前锁定</h3><span class="count">${locks.length}</span></div>`);
  if (!locks.length) {
    parts.push(`<p class="empty">还没有锁定。选中画布上的检测点，或直接在上面选择。</p>`);
  } else {
    parts.push(
      locks
        .map(
          (lk) => `<div class="lock-line">
            <span class="mono">${esc(lk.id)}</span>
            <span class="mono">${esc(lk.det_id)}</span>
            <span>→</span>
            <span class="mono">${esc(lk.track_id)}</span>
            <span class="muted">第 ${lk.frame} 帧</span>
            ${lk.note ? `<span class="small">${esc(lk.note)}</span>` : ""}
            <button class="btn-xs danger" data-unlock="${esc(lk.id)}" style="margin-left:auto">解除</button>
          </div>`
        )
        .join("")
    );
  }

  const conflicts = ov.conflicts || [];
  parts.push(`<div class="section-head"><h3>锁定冲突</h3><span class="count">${conflicts.length}</span></div>`);
  if (!conflicts.length) {
    parts.push(`<p class="empty">没有冲突。</p>`);
  } else {
    parts.push(
      conflicts
        .map(
          (c) => `<div class="lock-line conflict">
            <span class="tag">${esc(c.kind)}</span>
            <span>第 ${c.frame} 帧</span>
            ${c.det_id ? `<span class="mono">检测点 ${esc(c.det_id)}</span>` : ""}
            ${c.track_id ? `<span class="mono">轨迹 ${esc(c.track_id)}</span>` : ""}
            <span class="small">${esc(c.detail)}</span>
          </div>`
        )
        .join("")
    );
  }

  const revisions = ov.revisions || [];
  if (revisions.length) {
    parts.push(`<div class="section-head"><h3>锁定引发的已封存帧变更</h3><span class="count">${revisions.length}</span></div>`);
    parts.push(
      revisions
        .slice(-12)
        .reverse()
        .map(
          (r) => `<div class="lock-line correction">
            <span class="mono">第 ${r.frame} 帧</span>
            <span class="small">${
              r.before && Object.keys(r.before).length
                ? `原：${esc(Object.entries(r.before).map(([t, d]) => `${t}→${d}`).join(" "))}`
                : "原：未关联"
            }</span>
            <span class="small">→ 现：${esc(Object.entries(r.after || {}).map(([t, d]) => `${t}→${d}`).join(" ") || "未关联")}</span>
          </div>`
        )
        .join("")
    );
    parts.push(`<p class="panel-note">这些是新增的更正记录（append-only），不会抹掉旧结论。</p>`);
  }

  panel.innerHTML = parts.join("");

  const addBtn = $("#lock-add", panel);
  if (addBtn) addBtn.addEventListener("click", addLock);
  for (const btn of $$("[data-unlock]", panel)) {
    btn.addEventListener("click", () => removeLock(btn.dataset.unlock));
  }
  const frameInput = $("#lock-frame", panel);
  if (frameInput) {
    frameInput.addEventListener("change", async () => {
      const f = parseInt(frameInput.value, 10);
      if (Number.isNaN(f) || f < 0 || f > maxFrame()) return;
      await go(f);
      S.lockPrefill = { frame: f, detId: null };
      selectTab("locks");
      renderPanels();
    });
  }
}

async function addLock() {
  const frame = parseInt($("#lock-frame").value, 10);
  const detId = $("#lock-det").value;
  const trackId = $("#lock-track").value;
  const note = $("#lock-note").value || "";
  if (Number.isNaN(frame) || !detId || !trackId) {
    toast("请选择帧、检测点和轨迹", "crit");
    return;
  }
  try {
    const res = await API.post(`/api/sessions/${S.sessionId}/locks`, {
      frame,
      det_id: detId,
      track_id: trackId,
      note,
    });
    if (!res.ok) {
      toast(res.error || "锁定失败", "crit");
      return;
    }
    S.frameCache.clear();
    S.altCache.clear();
    S.lockPrefill = null;
    await reloadOverview();
    const changed = (res.changed_frames || []).length;
    const conflicts = (res.conflicts || []).length;
    toast(
      conflicts
        ? `锁定已加入，但有 ${conflicts} 处冲突：${res.conflicts.map((c) => c.det_id || c.track_id).filter(Boolean).join(", ")}`
        : changed
        ? `锁定已加入：${changed} 个已封存帧的关联被重算`
        : "锁定已加入，后续假设都已纳入该约束",
      conflicts ? "crit" : ""
    );
  } catch (err) {
    toast(`锁定失败：${err.message}`, "crit");
  }
}

async function removeLock(lockId) {
  try {
    const res = await API.del(`/api/sessions/${S.sessionId}/locks`, { lock_id: lockId });
    if (!res.ok) {
      toast(res.error || "解除失败", "crit");
      return;
    }
    S.frameCache.clear();
    S.altCache.clear();
    await reloadOverview();
    toast(`已解除 ${lockId}`);
  } catch (err) {
    toast(`解除失败：${err.message}`, "crit");
  }
}

async function reloadOverview() {
  const data = await API.get(`/api/sessions/${S.sessionId}`);
  await absorb(data.overview);
}

/* ---------- ledger panel ---------- */
function renderLedger(panel) {
  const ov = S.overview;
  const view = S.view;
  const parts = [];
  const pending = ov.pending_frames || [];
  parts.push(`<h3 style="margin-top:0">窗口状态</h3>`);
  parts.push(`<dl class="kv">
    <dt>已封存到</dt><dd>第 ${ov.sealed_until} 帧</dd>
    <dt>当前帧</dt><dd>第 ${view.frame} 帧 · ${view.tracker && view.tracker.sealed ? "已封存" : "窗口内（暂定）"}</dd>
    <dt>挂起帧</dt><dd>${pending.length ? pending.map((f) => `第 ${f} 帧`).join("、") : "无"}</dd>
    <dt>模式</dt><dd>${ov.mode === "window" ? "窗口推断" : "局部贪心"}</dd>
  </dl>`);
  parts.push(
    `<p class="panel-note">封存线之前的帧不再重算；迟到数据只追加“更正关系”，不会改写已封存轨迹。封存线以内的迟到帧会触发回滚重放。</p>`
  );

  const replays = ov.replays || [];
  parts.push(`<div class="section-head"><h3>回溯重放</h3><span class="count">${replays.length}</span></div>`);
  parts.push(
    replays.length
      ? replays
          .map(
            (r) => `<div class="lock-line replay">
              <span class="mono">第 ${r.frame} 帧</span>
              <span class="small">第 ${r.step} 步送达 · 回滚到第 ${r.horizon} 帧后重放</span>
              <div class="small">${esc(r.detail)}</div>
              ${(r.restored_hypotheses || []).length ? `<div class="small mono">恢复假设 ${esc(r.restored_hypotheses.join(", "))}</div>` : ""}
            </div>`
          )
          .join("")
      : `<p class="empty">没有重放。重放只在迟到帧仍落在开放窗口内时发生。</p>`
  );

  const corrections = ov.corrections || [];
  parts.push(`<div class="section-head"><h3>追加的更正</h3><span class="count">${corrections.length}</span></div>`);
  parts.push(
    corrections.length
      ? corrections
          .map(
            (c) => `<div class="lock-line correction">
              <span class="tag">${esc(c.kind)}</span>
              <span class="mono">第 ${c.frame} 帧 · ${esc(c.det_id || "")}</span>
              <div class="small">${esc(c.detail)}</div>
            </div>`
          )
          .join("")
      : `<p class="empty">没有更正。已封存帧的关联保持原样。</p>`
  );

  const revisions = ov.revisions || [];
  parts.push(`<div class="section-head"><h3>锁定重算记录</h3><span class="count">${revisions.length}</span></div>`);
  if (!revisions.length) {
    parts.push(`<p class="empty">没有因锁定重算而变化的已封存帧。</p>`);
  } else {
    parts.push(
      `<table><thead><tr><th>帧</th><th>之前</th><th>之后</th></tr></thead><tbody>${revisions
        .slice()
        .reverse()
        .map(
          (r) => `<tr><td class="mono">${r.frame}</td>
            <td class="mono small">${esc(Object.entries(r.before || {}).map(([t, d]) => `${t}→${d}`).join(" ") || "—")}</td>
            <td class="mono small">${esc(Object.entries(r.after || {}).map(([t, d]) => `${t}→${d}`).join(" ") || "—")}</td></tr>`
        )
        .join("")}</tbody></table>`
    );
  }

  const events = (ov.prune_events || []).filter((ev) => ev.dropped && ev.dropped.length);
  const dropped = events.reduce((acc, ev) => acc + ev.dropped.length, 0);
  parts.push(`<div class="section-head"><h3>被裁剪的假设</h3><span class="count">${dropped} 个 / ${events.length} 帧</span></div>`);
  if (!events.length) {
    parts.push(`<p class="empty">没有触发裁剪。</p>`);
  } else {
    parts.push(
      `<p class="panel-note">每个被丢弃假设都保留了相对该帧最优假设的代价差；“超出 beam 宽度”是预算裁剪，不代表一定更差。</p>`
    );
    parts.push(
      events
        .slice()
        .reverse()
        .slice(0, 14)
        .map(
          (ev) => `<div class="card">
            <div class="alt-head"><span class="mono">第 ${ev.frame} 帧</span><span class="alt-score">最优代价 ${fmt(ev.best_score, 2)}</span></div>
            <table><thead><tr><th>假设</th><th>代价差</th><th>原因</th></tr></thead><tbody>${ev.dropped
              .map(
                (d) => `<tr><td class="mono">${esc(d.key)}</td><td class="mono">+${fmt(d.gap, 2)}</td><td class="small">${esc(d.reason)}</td></tr>`
              )
              .join("")}</tbody></table>
          </div>`
        )
        .join("")
    );
    if (events.length > 14) parts.push(`<p class="small">仅显示最近 14 个裁剪事件，完整记录见导出 JSON 的 events。</p>`);
  }

  const lateNotes = ov.late_notes || [];
  if (lateNotes.length) {
    parts.push(`<div class="section-head"><h3>迟到帧说明</h3><span class="count">${lateNotes.length}</span></div>`);
    parts.push(
      lateNotes
        .map(
          (n) => `<div class="lock-line correction"><span class="mono">第 ${n.frame} 帧</span><span class="small">${esc(n.detail)}</span></div>`
        )
        .join("")
    );
  }

  panel.innerHTML = parts.join("");
}

/* ---------- metrics panel ---------- */
function renderMetrics(panel) {
  const ov = S.overview;
  const ev = ov.evaluation || {};
  const ident = ev.identity || {};
  const cfg = ov.config;
  const parts = [];

  parts.push(`<h3 style="margin-top:0">目标标识连续性</h3>`);
  parts.push(`<div class="stat-row">
    <div><div class="big-num ${ident.idsw ? "delta-bad" : "delta-good"}">${ident.idsw == null ? "—" : ident.idsw}</div><div class="big-label">身份切换 IDSW</div></div>
    <div><div class="big-num">${ident.clean_ratio == null ? "—" : fmt(ident.clean_ratio * 100, 0) + "%"}</div><div class="big-label">关联连续性</div></div>
    <div><div class="big-num ${ev.ghost_count ? "delta-bad" : "delta-good"}">${ev.ghost_count == null ? "—" : ev.ghost_count}</div><div class="big-label">虚警轨迹</div></div>
  </div>`);
  parts.push(
    `<p class="panel-note">只在已确认轨迹上统计身份切换；未确认的瞬时轨迹单独计为“试建关联”，不算身份错误。</p>`
  );

  const perTruth = ident.per_truth || {};
  const keys = Object.keys(perTruth).sort();
  if (keys.length) {
    parts.push(`<table><thead><tr><th>真值目标</th><th>命中</th><th>占用轨迹</th><th>连续性</th><th>切换</th></tr></thead><tbody>${keys
      .map((k) => {
        const p = perTruth[k];
        return `<tr>
          <td class="mono">${esc(k)}</td>
          <td class="mono">${p.detections}</td>
          <td class="mono">${esc((p.tracks || []).join(", "))}</td>
          <td class="mono ${p.continuity === 1 ? "" : "delta-bad"}">${p.continuity == null ? "—" : fmt(p.continuity * 100, 0) + "%"}</td>
          <td class="mono">${(p.switches || []).length}</td>
        </tr>`;
      })
      .join("")}</tbody></table>`);
  }

  const switches = ident.switches || [];
  if (switches.length) {
    parts.push(`<h3>切换明细</h3>`);
    parts.push(
      switches
        .map(
          (s) => `<div class="finding crit">真值 ${esc(s.truth)} · 第 ${s.frame} 帧：${esc(s.from_track)} → ${esc(s.to_track)}（间隔 ${s.gap_frames} 帧）</div>`
        )
        .join("")
    );
  }

  const frag = ident.fragmentation || {};
  parts.push(`<h3>轨迹分段</h3>`);
  const fragRows = Object.entries(frag);
  parts.push(
    fragRows.length
      ? `<table><thead><tr><th>真值</th><th>轨迹数</th></tr></thead><tbody>${fragRows
          .map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="mono ${v > 1 ? "delta-bad" : ""}">${v}</td></tr>`)
          .join("")}</tbody></table>`
      : `<p class="empty">没有可统计的目标。</p>`
  );

  const ghosts = ev.ghosts || [];
  parts.push(`<h3>虚警轨迹</h3>`);
  parts.push(
    ghosts.length
      ? ghosts
          .map(
            (g) => `<div class="finding warn"><span class="mono">${esc(g.track)}</span> · ${g.detections} 次关联 · 出生第 ${g.birth_frame} 帧 · ${esc((g.det_ids || []).join(", "))}</div>`
          )
          .join("")
      : `<p class="empty">没有形成持续虚警的轨迹。</p>`
  );

  const br = ev.branching || {};
  parts.push(`<h3>分支恢复</h3>`);
  parts.push(`<dl class="kv">
    <dt>出现多假设的帧</dt><dd>${br.frames_with_alternatives == null ? "—" : br.frames_with_alternatives}</dd>
    <dt>记录了不同身份假设</dt><dd>${br.frames_with_recorded_variant == null ? "—" : br.frames_with_recorded_variant}</dd>
    <dt>回溯重放</dt><dd>${(ev.replays || []).length}</dd>
    <dt>追加更正</dt><dd>${(ev.corrections || []).length}</dd>
    <dt>迟到帧</dt><dd>${(ev.late_frames || []).join(", ") || "无"}</dd>
  </dl>`);
  const details = (br.details || []).filter((d) => d.identity_variant_recorded).slice(-10);
  if (details.length) {
    parts.push(
      `<table><thead><tr><th>帧</th><th>假设数</th><th>与 MAP 的差</th><th>身份变体</th></tr></thead><tbody>${details
        .map(
          (d) => `<tr><td class="mono">${d.frame}</td><td class="mono">${d.alternatives}</td><td class="mono">${fmt(d.committed_gap_to_map, 2)}</td><td class="small mono">${esc(d.variant_keys.join(", "))}</td></tr>`
        )
        .join("")}</tbody></table>`
    );
  }
  parts.push(
    `<p class="panel-note">“分支恢复”指：封存时记录了与 MAP 不同的具体身份假设（及它的代价差），即使它被裁剪，也能回看它为什么输。</p>`
  );

  parts.push(`<h3>搜索规模与预算</h3>`);
  parts.push(`<dl class="kv">
    <dt>窗口长度</dt><dd>${cfg.window_len} 帧</dd>
    <dt>beam 宽度</dt><dd>${cfg.beam_width}</dd>
    <dt>每帧 k-best</dt><dd>${cfg.k_best}</dd>
    <dt>节点预算</dt><dd>${cfg.kbest_node_budget}</dd>
    <dt>是否截断</dt><dd>${ov.truncated_search ? "是（结果非全局最优）" : "否"}</dd>
  </dl>`);
  parts.push(
    `<p class="panel-note crit">本工具从不声称全局最优：窗口推断在有限 beam 上比较假设，超出预算的假设被裁剪并留档。</p>`
  );

  parts.push(`<h3>代价模型</h3>`);
  parts.push(`<dl class="kv">
    <dt>门控</dt><dd>χ² ≤ ${fmt(cfg.gate_chi2, 1)}</dd>
    <dt>检测概率 Pd</dt><dd>${fmt(cfg.pd, 2)}</dd>
    <dt>杂点密度</dt><dd>${cfg.clutter_density}</dd>
    <dt>漏检代价</dt><dd>${fmt(cfg.miss_cost)}</dd>
    <dt>新生代价</dt><dd>${fmt(cfg.birth_cost)}</dd>
    <dt>确认 (M/N)</dt><dd>${cfg.confirm_hits} / ${cfg.confirm_window}</dd>
  </dl>`);

  parts.push(`<h3>${S.mode === "window" ? "搜索预算（可调）" : "贪心参数（可调）"}</h3>`);
  parts.push(
    S.mode === "window"
      ? `<p class="panel-note">调小预算会更快、裁剪更多；每次裁剪都留档。修改后会以相同锁定重新计算整个会话。</p>`
      : `<p class="panel-note">贪心模式没有窗口/beam；这里只能调整门控。窗口与 beam 参数在切回窗口推断后生效。</p>`
  );
  const fields = S.mode === "window"
    ? [
        ["window_len", "窗口帧数", 1, 30, 1],
        ["beam_width", "beam 宽度", 1, 20, 1],
        ["k_best", "每帧 k-best", 1, 12, 1],
        ["branch_margin", "分支阈值", 0.1, 20, 0.1],
        ["prune_margin", "保留阈值", 0.1, 60, 0.1],
        ["gate_chi2", "门控 χ²", 1, 100, 1],
      ]
    : [["gate_chi2", "门控 χ²", 1, 100, 1]];
  parts.push(
    `<div class="lock-form">${fields
      .map(
        ([key, label, min, max, step]) =>
          `<label class="small">${label}
            <input id="cfg-${key}" type="number" min="${min}" max="${max}" step="${step}" value="${cfg[key]}" style="width:74px">
          </label>`
      )
      .join("")}<button class="primary" id="cfg-apply" type="button">应用</button></div>`
  );

  parts.push(`<div class="btn-row"><button class="btn-xs" id="metrics-compare">与另一种模式对比</button><button class="btn-xs" id="metrics-export">导出 JSON</button></div>`);

  panel.innerHTML = parts.join("");
  const cmpBtn = $("#metrics-compare", panel);
  if (cmpBtn) cmpBtn.addEventListener("click", openCompare);
  const expBtn = $("#metrics-export", panel);
  if (expBtn) expBtn.addEventListener("click", exportJson);
  const applyBtn = $("#cfg-apply", panel);
  if (applyBtn) applyBtn.addEventListener("click", applyConfig);
}

async function applyConfig() {
  const overrides = {};
  for (const key of ["window_len", "beam_width", "k_best", "branch_margin", "prune_margin", "gate_chi2"]) {
    const input = $(`#cfg-${key}`);
    if (!input) continue;
    const val = parseFloat(input.value);
    if (!Number.isFinite(val)) continue;
    overrides[key] = ["window_len", "beam_width", "k_best"].includes(key) ? Math.round(val) : val;
  }
  try {
    const data = await API.post(`/api/sessions/${S.sessionId}/rebuild`, {
      mode: S.mode,
      overrides,
    });
    S.frameCache.clear();
    S.altCache.clear();
    await absorb(data.overview);
    toast("参数已应用；已封存帧若因锁定/参数变化而不同，会记入重算记录");
  } catch (err) {
    toast(`应用参数失败：${err.message}`, "crit");
  }
}

/* ---------- compare ---------- */
async function openCompare() {
  const dialog = $("#compare-dialog");
  const body = $("#compare-body");
  const scenarioKey = (S.overview && S.overview.scenario.key) || S.scenario;
  body.innerHTML = `<p class="empty">正在对比 ${esc(scenarioKey)} …</p>`;
  if (typeof dialog.showModal === "function") {
    if (!dialog.open) dialog.showModal();
  } else {
    dialog.setAttribute("open", "");
  }
  try {
    const data = await API.get(`/api/compare?scenario=${encodeURIComponent(scenarioKey)}`);
    body.innerHTML = renderCompare(data);
  } catch (err) {
    body.innerHTML = `<p class="finding crit">对比失败：${esc(err.message)}</p>`;
  }
}

function renderCompare(data) {
  const g = data.greedy && data.greedy.evaluation;
  const w = data.window && data.window.evaluation;
  if (!g || !w) return `<p class="empty">没有对比数据。</p>`;
  const delta = data.delta || {};
  const col = (ev, cls, title) => `
    <div class="cmp-col ${cls}">
      <h4>${title}</h4>
      <dl class="kv">
        <dt>IDSW</dt><dd>${ev.identity.idsw}</dd>
        <dt>连续性</dt><dd>${fmt(ev.identity.clean_ratio * 100, 0)}%</dd>
        <dt>虚警</dt><dd>${ev.ghost_count}</dd>
        <dt>多假设帧</dt><dd>${ev.branching.frames_with_alternatives}</dd>
        <dt>重放 / 更正</dt><dd>${ev.replays.length} / ${ev.corrections.length}</dd>
      </dl>
      <div class="small" style="margin-top:8px">分段：${esc(
        Object.entries(ev.identity.fragmentation || {}).map(([k, v]) => `${k}×${v}`).join(" ")
      ) || "—"}</div>
    </div>`;
  const deltaText = (val, lowerBetter) => {
    if (typeof val !== "number" || val === 0) return `<span class="muted">0</span>`;
    const good = lowerBetter ? val < 0 : val > 0;
    return `<span class="${good ? "delta-good" : "delta-bad"}">${val > 0 ? "+" : ""}${val}</span>`;
  };
  return `
    <p class="panel-note">同一场景、同一配置、同一随机种子，只改变关联策略。数字来自真值评估（追踪器本身不可见）。</p>
    <div class="cmp-grid">
      ${col(g, "g", "局部贪心（逐帧最近优先，立即提交）")}
      ${col(w, "w", "窗口推断（多假设 + 有限预算）")}
    </div>
    <h3>差异（窗口 − 贪心）</h3>
    <table><thead><tr><th>指标</th><th>贪心</th><th>窗口</th><th>差值</th></tr></thead><tbody>
      <tr><td>身份切换 IDSW（越低越好）</td><td class="mono">${g.identity.idsw}</td><td class="mono">${w.identity.idsw}</td><td class="mono">${deltaText(delta.idsw, true)}</td></tr>
      <tr><td>连续性（越高越好）</td><td class="mono">${fmt(g.identity.clean_ratio * 100, 0)}%</td><td class="mono">${fmt(w.identity.clean_ratio * 100, 0)}%</td><td class="mono">${deltaText(Math.round((delta.clean_ratio || 0) * 1000) / 10, false)}</td></tr>
      <tr><td>虚警轨迹（越低越好）</td><td class="mono">${g.ghost_count}</td><td class="mono">${w.ghost_count}</td><td class="mono">${deltaText(delta.ghosts, true)}</td></tr>
      <tr><td>多假设帧（记录量）</td><td class="mono">${g.branching.frames_with_alternatives}</td><td class="mono">${w.branching.frames_with_alternatives}</td><td class="mono">${deltaText(delta.branch_frames, false)}</td></tr>
      <tr><td>回溯重放（记录量）</td><td class="mono">${g.replays.length}</td><td class="mono">${w.replays.length}</td><td class="mono">${deltaText(delta.replays, false)}</td></tr>
      <tr><td>追加更正（记录量）</td><td class="mono">${g.corrections.length}</td><td class="mono">${w.corrections.length}</td><td class="mono">${deltaText(delta.corrections, false)}</td></tr>
    </tbody></table>
    <p class="panel-note" style="margin-top:10px">窗口推断的“多假设帧”与“重放/更正”不是错误，而是它保留并记录歧义的代价：这些记录让“另一条轨迹可能才是对的”可被回看。</p>
    <p class="panel-note">两种模式都不是全局最优：贪心没有回溯；窗口推断的假设在 beam/预算之外被明确裁剪。</p>
  `;
}

/* ---------- export ---------- */
async function exportJson() {
  if (!S.sessionId) return;
  try {
    const data = await API.get(`/api/sessions/${S.sessionId}/export`);
    const blob = new Blob([JSON.stringify(data.data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const key = (S.overview && S.overview.scenario.key) || S.scenario;
    a.download = `association-${key}-${S.mode}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
    toast("已导出原始检测与关联结果（JSON）");
  } catch (err) {
    toast(`导出失败：${err.message}`, "crit");
  }
}

/* ---------- go ---------- */
boot();
