/* Headless smoke harness: runs web/app.js against a live server with a tiny
 * fake DOM.  It does not render pixels; it proves the data-shape assumptions
 * in every panel renderer and that the event wiring doesn't throw.
 *
 * usage: node tests/headless_ui.mjs http://127.0.0.1:PORT
 */
import fs from "node:fs";
import vm from "node:vm";

const BASE = process.argv[2] || "http://127.0.0.1:8810";
const SRC = fs.readFileSync(new URL("../web/app.js", import.meta.url), "utf8");
const HTML = fs.readFileSync(new URL("../web/index.html", import.meta.url), "utf8");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---- collect ids the real page defines, so missing ids surface as bugs ---- */
const ids = new Set([...HTML.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
const dynamicIds = new Set([
  "chart-tip", "det-clear", "det-lock-btn", "lock-add", "lock-det", "lock-frame",
  "lock-note", "lock-track", "metrics-compare", "metrics-export", "truth-toggle",
  "alt-slot-any",
]);
const created = [];

class ClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(...c) { c.forEach((x) => this.set.add(x)); }
  remove(...c) { c.forEach((x) => this.set.delete(x)); }
  toggle(c, force) {
    const on = force === undefined ? !this.set.has(c) : !!force;
    if (on) this.set.add(c); else this.set.delete(c);
    return on;
  }
  contains(c) { return this.set.has(c); }
  get value() { return [...this.set].join(" "); }
}

class Element {
  constructor(tag, id) {
    this.tagName = (tag || "div").toUpperCase();
    this.id = id || "";
    this.dataset = {};
    this.classList = new ClassList(this);
    this.children = [];
    this.style = {};
    this.handlers = {};
    this.className = "";
    this.textContent = "";
    this._html = "";
    this.value = "";
    this.checked = false;
    this.width = 760;
    this.height = 760;
  }
  set innerHTML(v) { this._html = String(v); }
  get innerHTML() { return this._html; }
  setAttribute(k, v) { this[k] = v; if (k === "id") this.id = v; }
  getAttribute(k) { return this[k]; }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  removeEventListener() {}
  appendChild(child) { this.children.push(child); return child; }
  remove() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: 760, height: 760, right: 760, bottom: 760 }; }
  getContext() { return ctx2d; }
  focus() {}
  closest() { return null; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  fire(type, ev = {}) {
    for (const fn of this.handlers[type] || []) fn({ target: this, ...ev });
  }
}

const ctx2d = new Proxy(
  { canvas: null, setTransform() {}, clearRect() {}, beginPath() {}, moveTo() {},
    lineTo() {}, stroke() {}, arc() {}, fill() {}, fillText() {}, strokeRect() {},
    save() {}, restore() {}, setLineDash() {} },
  {
    get(t, k) { return k in t ? t[k] : undefined; },
    set(t, k, v) { t[k] = v; return true; },
  }
);

const registry = new Map();
function el(selector) {
  if (!registry.has(selector)) {
    let tag = "div";
    if (/canvas/.test(selector)) tag = "canvas";
    if (/select/.test(selector)) tag = "select";
    if (/input|scrub|toggle/.test(selector)) tag = "input";
    if (/btn|mode-|tab/.test(selector)) tag = "button";
    registry.set(selector, new Element(tag, selector.replace(/^[#.]/, "")));
  }
  return registry.get(selector);
}

const byId = new Map();
for (const id of ids) byId.set(id, new Element("div", id));

const document = {
  documentElement: {},
  body: new Element("body"),
  querySelector(sel) {
    if (sel.startsWith("#")) {
      const id = sel.slice(1);
      if (!byId.has(id)) {
        if (!dynamicIds.has(id) && !id.startsWith("alt-slot-")) {
          created.push(id);
        }
        byId.set(id, new Element(id === "canvas" ? "canvas" : "div", id));
      }
      return byId.get(id);
    }
    return el(sel);
  },
  querySelectorAll(sel) {
    if (sel === ".tab") {
      return ["branches", "detections", "locks", "ledger", "metrics"].map((t) => {
        const e = new Element("button");
        e.dataset.tab = t;
        return e;
      });
    }
    if (sel === ".panel") {
      return ["branches", "detections", "locks", "ledger", "metrics"].map((t) =>
        document.querySelector(`#panel-${t}`)
      );
    }
    if (sel === "#timeline .seg") return [];
    if (sel === "[data-detail]" || sel === "[data-lock]" || sel === "[data-det]" ||
        sel === "[data-unlock]" || sel === "[data-pair-det]") return [];
    return [];
  },
  createElement(tag) { return new Element(tag); },
  addEventListener() {},
};

const timers = [];
const window = {
  addEventListener() {},
  devicePixelRatio: 1,
  setTimeout: (fn, ms) => { const h = setTimeout(fn, ms); timers.push(h); return h; },
  clearTimeout,
  setInterval: () => 0,
  clearInterval() {},
};

const sandbox = {
  document,
  window,
  console,
  fetch,
  Blob,
  URL,
  CSS: { escape: (s) => s },
  getComputedStyle: () => ({
    getPropertyValue: (name) => ({
      "--s1": "#3987e5", "--s2": "#d95926", "--s3": "#199e70",
      "--ink": "#f0efec", "--ink-2": "#c3c2b7", "--muted": "#898781",
      "--axis": "#383835", "--warning": "#fab219",
    }[name] || ""),
  }),
  setTimeout: window.setTimeout,
  clearTimeout,
  setInterval: () => 0,
  clearInterval() {},
  Math,
  JSON,
  Number,
  String,
  Object,
  Array,
  Set,
  Map,
  Promise,
  encodeURIComponent,
  parseInt,
  Infinity,
  NaN,
};

const failures = [];
process.on("unhandledRejection", (err) => failures.push(`unhandled: ${err && err.message}`));

const context = vm.createContext(sandbox);
const originalFetch = fetch;
sandbox.fetch = (path, options) => originalFetch(new URL(path, BASE), options);

try {
  vm.runInContext(SRC, context, { filename: "app.js" });
} catch (err) {
  console.error("LOAD ERROR", err);
  process.exit(1);
}

await sleep(2500);

function state(name) {
  try {
    return vm.runInContext(name, context);
  } catch (err) {
    failures.push(`${name.slice(0, 80)}: ${String(err.message).slice(0, 300)}`);
    return null;
  }
}

const S = state("S");
if (!S || !S.sessionId) {
  console.error("no session was created; failures:", failures);
  process.exit(1);
}
console.log("session", S.sessionId, "scenario", S.scenario, "frames", S.history.length);

/* drive through every frame, every tab, and every alternative */
const n = S.overview.scenario.n_frames;
const tabs = ["branches", "detections", "locks", "ledger", "metrics"];
for (let f = 0; f < n; f += 1) {
  await state(`go(${f})`);
  for (const tab of tabs) {
    await state(`selectTab(${JSON.stringify(tab)}); renderPanels()`);
  }
  const alts = (S.view && S.view.tracker && S.view.tracker.alternatives) || [];
  for (const a of alts.filter((x) => !x.is_map).slice(0, 2)) {
    await state(`showAltDetail(${JSON.stringify(a.key)})`);
    await sleep(30);
  }
}

/* exercise lock add/remove through the UI code path */
state("S.selectedDet = S.view.detections[0] && S.view.detections[0].id");
await state(`openLockForm(S.frame, S.selectedDet || S.view.detections[0].id)`);
// The fake <select> has no option list; give it the values the real page's
// options would supply (the panel renders <option>s from exactly these lists).
{
  const view = S.view;
  const frame = S.frame;
  const lf = document.querySelector("#lock-frame");
  const ld = document.querySelector("#lock-det");
  lf.value = String(frame);
  ld.value = view.detections[0].id;
  const node = (view.tracker && (view.tracker.committed || (view.tracker.alternatives || [])[0])) || {};
  document.querySelector("#lock-track").value = node.tracks?.[0]?.id || "";
  document.querySelector("#lock-note").value = "headless smoke";
}
await state(`addLock()`);
await sleep(700);
if (!S.overview.locks.length) failures.push("addLock did not register a lock");
const lockId = (S.overview.locks[0] || {}).id;
if (lockId) {
  await state(`removeLock(${JSON.stringify(lockId)})`);
  await sleep(600);
  if (S.overview.locks.length) failures.push("removeLock did not remove the lock");
}

/* compare dialog */
await state("openCompare()");
await sleep(1500);
const compareBody = byId.get("compare-body");
if (!/cmp-grid/.test(compareBody.innerHTML)) failures.push("compare dialog did not render");

/* mode switch */
await state("S.mode = 'greedy'");
const modeBtn = byId.get("mode-greedy");
try {
  modeBtn.fire("click");
} catch (err) {
  failures.push(`mode switch threw: ${err.message}`);
}
await sleep(2000);
if (S.mode !== "greedy") failures.push("mode switch did not take effect");

for (const field of ["#status-chips", "#legend", "#panel-metrics", "#panel-ledger"]) {
  const node = document.querySelector(field);
  if (node && !node.innerHTML) failures.push(`${field} rendered empty`);
}

const panelMetrics = byId.get("panel-metrics").innerHTML;
if (!/身份切换/.test(panelMetrics)) failures.push("metrics panel missing identity section");

/* per-scenario pass: hit every late/ambiguous frame and the alternative viewer */
const scenarios = ["crossing", "vanish", "clutter", "combined"];
for (const key of scenarios) {
  state(`S.scenario = ${JSON.stringify(key)}; S.mode = "window"; S.frame = 0`);
  const settled = await state(`createSession()`);
  await settled;
  await sleep(300);
  const frames = (S.overview && S.overview.scenario.n_frames) || 0;
  if (!frames) { failures.push(`${key}: could not create session`); continue; }
  for (let f = 0; f < frames; f += 1) {
    await state(`go(${f})`);
    for (const tab of tabs) await state(`selectTab(${JSON.stringify(tab)}); renderPanels()`);
    if (S.view && S.view.tracker && S.view.tracker.sealed_kind === "late_correction") {
      const html = byId.get("panel-branches").innerHTML;
      if (!/迟到送达/.test(html)) failures.push(`${key} f${f}: late-correction frame lacks explanation`);
      const ledger = byId.get("panel-ledger").innerHTML;
      if (!/更正/.test(ledger)) failures.push(`${key} f${f}: ledger lacks corrections`);
    }
    for (const a of ((S.view && S.view.tracker && S.view.tracker.alternatives) || []).filter((x) => !x.is_map).slice(0, 1)) {
      await state(`showAltDetail(${JSON.stringify(a.key)})`);
      await sleep(20);
      if (!(S.altDetail && S.altDetail.key === a.key)) {
        failures.push(`${key} f${f}: alternative detail missing for ${a.key}`);
      }
    }
  }
}
console.log("failures:", failures.length ? failures : "none");
process.exit(failures.length ? 1 : 0);
