"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  mode: "generate",
  options: null,
  health: {},
  sizeByMode: { generate: { tier: "2K", ratio: "1:1" }, edit: { tier: "auto", ratio: "1:1" } },
  inputs: [],            // [{ blob, url }]
  lastResult: null,
  pollTimer: null,
  busy: false,
  history: [],
  filters: { mode: "all", backend: "all" },
  detail: null,
};

// ---------- helpers ----------
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

const backend = () => $('input[name="backend"]:checked').value;
const outputUrl = (file) => `/outputs/${encodeURIComponent(file)}`;
const round16 = (v) => Math.round(v / 16) * 16;
const show = (el, on = true) => el.classList.toggle("hidden", !on);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "onclick") node.addEventListener("click", v);
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

// ---------- tabs & mode ----------
function switchTab(tab) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  const gallery = tab === "gallery";
  show($("#panel-create"), !gallery);
  show($("#panel-gallery"), gallery);
  if (gallery) loadHistory();
  else setMode(tab);
}

function setMode(mode) {
  state.mode = mode;
  document.body.dataset.mode = mode;
  $("#prompt-label").textContent = mode === "edit" ? "Instruction" : "Prompt";
  $("#prompt").placeholder = mode === "edit"
    ? "Change the background to a sunset beach  ·  Extract the person  ·  Put the cat from image 1 on the sofa in image 2"
    : "A cozy cabin in a snowy forest at dusk, warm light in the windows";
  $("#submit").textContent = mode === "edit" ? "Edit" : "Generate";
  renderSize();
  updateSubmitEnabled();
}

// ---------- size picker ----------
function renderSize() {
  const sel = state.sizeByMode[state.mode];
  $$("#tier button").forEach((b) => b.classList.toggle("active", b.dataset.tier === sel.tier));
  const ratios = $("#ratios");
  ratios.innerHTML = "";
  if (state.options && (sel.tier === "2K" || sel.tier === "1K")) {
    for (const p of state.options.sizes[sel.tier]) {
      const b = el("button", { type: "button", title: `${p.width}×${p.height}` }, p.label);
      b.classList.toggle("active", p.label === sel.ratio);
      b.addEventListener("click", () => { sel.ratio = p.label; renderSize(); });
      ratios.append(b);
    }
  }
  show($("#custom-size"), sel.tier === "custom");
}

function currentSize() {
  const sel = state.sizeByMode[state.mode];
  if (sel.tier === "auto") return "auto";
  if (sel.tier === "custom") return `${round16(+$("#cw").value)}x${round16(+$("#ch").value)}`;
  const p = state.options.sizes[sel.tier].find((x) => x.label === sel.ratio) || state.options.sizes[sel.tier][0];
  return `${p.width}x${p.height}`;
}

function applySize(mode, width, height) {
  const sel = state.sizeByMode[mode];
  for (const [tier, presets] of Object.entries(state.options.sizes)) {
    const p = presets.find((x) => x.width === width && x.height === height);
    if (p) { sel.tier = tier; sel.ratio = p.label; return; }
  }
  sel.tier = "custom";
  $("#cw").value = width;
  $("#ch").value = height;
}

// ---------- edit inputs ----------
function addBlobs(blobs) {
  const max = state.options ? state.options.limits.max_images : 10;
  for (const blob of blobs) {
    if (!blob.type.startsWith("image/")) continue;
    if (state.inputs.length >= max) { showError(`At most ${max} input images.`); break; }
    state.inputs.push({ blob, url: URL.createObjectURL(blob) });
  }
  renderThumbs();
}

async function addFromUrl(url) {
  const blob = await (await fetch(url)).blob();
  addBlobs([blob]);
}

function moveInput(i, delta) {
  const j = i + delta;
  if (j < 0 || j >= state.inputs.length) return;
  [state.inputs[i], state.inputs[j]] = [state.inputs[j], state.inputs[i]];
  renderThumbs();
}

function removeInput(i) {
  URL.revokeObjectURL(state.inputs[i].url);
  state.inputs.splice(i, 1);
  renderThumbs();
}

function renderThumbs() {
  const ul = $("#thumbs");
  ul.innerHTML = "";
  state.inputs.forEach((inp, i) => {
    ul.append(el("li", {},
      el("span", { class: "num" }, String(i + 1)),
      el("img", { src: inp.url, alt: `Input ${i + 1}` }),
      el("div", { class: "tools" },
        el("button", { type: "button", title: "Move left", onclick: () => moveInput(i, -1) }, "←"),
        el("button", { type: "button", title: "Remove", onclick: () => removeInput(i) }, "×"),
        el("button", { type: "button", title: "Move right", onclick: () => moveInput(i, 1) }, "→"))));
  });
  updateSubmitEnabled();
}

// ---------- health ----------
async function refreshHealth() {
  try {
    state.health = (await api("/api/health")).backends;
  } catch {
    state.health = {};
  }
  for (const name of ["local", "space"]) {
    const s = state.health[name];
    $(`#dot-${name}`).className = `dot ${s ? s.state : ""}`;
    $(`#dot-${name}`).title = s ? `${s.state}: ${s.detail}` : "unreachable";
  }
  const s = state.health[backend()];
  const banner = $("#banner");
  if (s && s.state !== "ready") {
    banner.textContent = s.state === "loading"
      ? "Loading Qwen-Image-2.1 on your GPU… (the first run downloads the weights). You can use HF Space meanwhile."
      : `${backend() === "local" ? "Local GPU" : "HF Space"} unavailable — ${s.detail}`;
    banner.classList.toggle("error", s.state === "error");
    show(banner, true);
  } else {
    show(banner, false);
  }
  updateSubmitEnabled();
}

function updateSubmitEnabled() {
  const s = state.health[backend()];
  const ready = s && s.state === "ready";
  const needsInputs = state.mode === "edit" && state.inputs.length === 0;
  $("#submit").disabled = state.busy || !ready || needsInputs;
}

// ---------- messages ----------
function showError(msg) { $("#error").textContent = msg; show($("#error"), true); }
function showWarning(msg) { $("#warning").textContent = msg; show($("#warning"), !!msg); }
function clearMessages() { show($("#error"), false); show($("#warning"), false); }

// ---------- jobs ----------
async function submitJob(ev) {
  ev.preventDefault();
  if ($("#submit").disabled) return;
  const prompt = $("#prompt").value.trim();
  if (!prompt) return showError("Please enter a prompt.");

  const fd = new FormData();
  fd.append("backend", backend());
  fd.append("prompt", prompt);
  const neg = $("#negative").value;
  if (neg.trim()) fd.append("negative_prompt", neg);
  fd.append("size", currentSize());
  if (backend() === "local") fd.append("steps", $("#steps").value || "40");
  const seed = $("#seed").value.trim();
  if (seed) fd.append("seed", seed);
  fd.append("transparent", $("#transparent").checked ? "true" : "false");
  fd.append("enhance", $("#enhance").checked ? "true" : "false");
  if (state.mode === "edit") {
    state.inputs.forEach((inp, i) => fd.append("images", inp.blob, `input-${i}.png`));
  }

  clearMessages();
  setBusy(true);
  setProgress({ status: "queued", queue_position: null });
  try {
    const { job_id } = await api("/api/jobs", { method: "POST", body: fd });
    pollJob(job_id);
  } catch (err) {
    setBusy(false);
    show($("#progress"), false);
    showError(err.message);
  }
}

function setBusy(busy) {
  state.busy = busy;
  updateSubmitEnabled();
}

function setProgress(job) {
  show($("#progress"), true);
  const bar = $(".bar");
  const fill = $("#bar-fill");
  let text = "Working…";
  let pct = null;
  if (job.status === "queued") {
    text = job.queue_position ? `Queued (#${job.queue_position})` : "Queued…";
  } else if (job.phase === "enhancing") {
    text = "Enhancing prompt…";
  } else if (job.phase === "generating" && job.total_steps) {
    text = `Step ${job.step}/${job.total_steps}`;
    pct = (100 * job.step) / job.total_steps;
  } else if (job.phase === "remote-queue") {
    text = job.step ? `Waiting on HF Space (#${job.step})` : "Running on HF Space…";
  }
  $("#progress-text").textContent = text;
  bar.classList.toggle("indeterminate", pct === null);
  fill.style.width = pct === null ? "" : `${pct}%`;
}

function pollJob(jobId) {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (err) {
      clearInterval(state.pollTimer);
      setBusy(false);
      show($("#progress"), false);
      return showError(err.message);
    }
    if (job.status === "done") {
      clearInterval(state.pollTimer);
      setBusy(false);
      show($("#progress"), false);
      showResult(job.result);
      showWarning(job.warning);
    } else if (job.status === "error") {
      clearInterval(state.pollTimer);
      setBusy(false);
      show($("#progress"), false);
      showError(job.error);
    } else {
      setProgress(job);
    }
  }, 1000);
}

function showResult(meta) {
  state.lastResult = meta;
  show($("#result-empty"), false);
  show($("#result-figure"), true);
  $("#result-img").src = `${outputUrl(meta.file)}?t=${Date.now()}`;
  $("#result-frame").classList.toggle("checker", meta.transparent);
  $("#download").href = outputUrl(meta.file);
  $("#download").download = meta.file;
  const inputs = $("#result-inputs");
  inputs.innerHTML = "";
  meta.input_files.forEach((f) => inputs.append(el("img", { src: outputUrl(f), alt: "Input" })));
  show($("#rewritten"), !!meta.rewritten_prompt);
  $("#rewritten-text").textContent = meta.rewritten_prompt || "";
}

// ---------- reuse / edit this ----------
async function reuseSettings(meta) {
  switchTab(meta.mode);
  $(`input[name="backend"][value="${meta.backend}"]`).checked = true;
  onBackendChange();
  $("#prompt").value = meta.prompt;
  $("#negative").value = meta.negative_prompt.trim();
  $("#seed").value = meta.seed;
  if (meta.steps) $("#steps").value = meta.steps;
  $("#transparent").checked = meta.transparent;
  $("#enhance").checked = meta.enhance;
  applySize(meta.mode, meta.width, meta.height);
  if (meta.mode === "edit") {
    state.inputs.forEach((inp) => URL.revokeObjectURL(inp.url));
    state.inputs = [];
    for (const f of meta.input_files) await addFromUrl(outputUrl(f));
  }
  renderSize();
  renderThumbs();
}

async function editThis(meta) {
  switchTab("edit");
  await addFromUrl(outputUrl(meta.file));
}

// ---------- gallery ----------
async function loadHistory() {
  try {
    state.history = await api("/api/history");
  } catch (err) {
    state.history = [];
  }
  renderGrid();
}

function renderGrid() {
  const grid = $("#grid");
  grid.innerHTML = "";
  const items = state.history.filter((m) =>
    (state.filters.mode === "all" || m.mode === state.filters.mode) &&
    (state.filters.backend === "all" || m.backend === state.filters.backend));
  for (const m of items) {
    grid.append(el("div", { class: "card", onclick: () => openDetail(m) },
      el("img", { src: outputUrl(m.file), loading: "lazy", alt: m.prompt }),
      el("p", { title: m.prompt }, `${m.mode === "edit" ? "✎ " : ""}${m.prompt}`)));
  }
  show($("#grid-empty"), items.length === 0);
}

function openDetail(meta) {
  state.detail = meta;
  $("#detail-img").src = outputUrl(meta.file);
  $("#detail-frame").classList.toggle("checker", meta.transparent);
  $("#d-download").href = outputUrl(meta.file);
  $("#d-download").download = meta.file;
  const inputs = $("#detail-inputs");
  inputs.innerHTML = "";
  meta.input_files.forEach((f) => inputs.append(el("img", { src: outputUrl(f), alt: "Input" })));
  const rows = [
    ["Prompt", meta.prompt],
    ["Enhanced", meta.rewritten_prompt],
    ["Negative", meta.negative_prompt.trim()],
    ["Mode", meta.mode],
    ["Backend", meta.backend],
    ["Size", `${meta.width}×${meta.height} (${meta.size_label})`],
    ["Steps", meta.steps],
    ["Seed", meta.seed],
    ["Transparent", meta.transparent ? "yes" : "no"],
    ["Time", `${meta.duration_s}s`],
    ["Created", new Date(meta.created_at).toLocaleString()],
  ];
  const dl = $("#detail-dl");
  dl.innerHTML = "";
  for (const [k, v] of rows) {
    if (v === null || v === undefined || v === "") continue;
    dl.append(el("dt", {}, k), el("dd", {}, String(v)));
  }
  $("#detail").showModal();
}

async function deleteDetail() {
  const meta = state.detail;
  if (!meta || !confirm("Delete this image?")) return;
  await api(`/api/history/${meta.id}`, { method: "DELETE" });
  $("#detail").close();
  loadHistory();
}

// ---------- backend switch ----------
function onBackendChange() {
  document.body.dataset.backend = backend();
  refreshHealth();
}

// ---------- wiring ----------
function wire() {
  $$(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  $$('input[name="backend"]').forEach((r) => r.addEventListener("change", onBackendChange));
  $$("#tier button").forEach((b) => b.addEventListener("click", () => {
    state.sizeByMode[state.mode].tier = b.dataset.tier;
    renderSize();
  }));
  $("#job-form").addEventListener("submit", submitJob);
  $("#prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submitJob(e);
  });
  // Random seeds are capped at 2^31-1 (not 2^32-1) so a dice roll works on
  // both the local backend (max 2^32-1) and the HF Space (max 2^31-1).
  $("#dice").addEventListener("click", () => { $("#seed").value = Math.floor(Math.random() * 2 ** 31); });

  const dz = $("#dropzone");
  $("#pick").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (e) => { addBlobs(e.target.files); e.target.value = ""; });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); addBlobs(e.dataTransfer.files); });
  document.addEventListener("paste", (e) => {
    if (state.mode !== "edit") return;
    const files = Array.from(e.clipboardData.files || []);
    if (files.length) addBlobs(files);
  });

  $("#use-rewritten").addEventListener("click", () => {
    $("#prompt").value = state.lastResult.rewritten_prompt;
    $("#enhance").checked = false;
  });
  $("#reuse").addEventListener("click", () => reuseSettings(state.lastResult));
  $("#edit-this").addEventListener("click", () => editThis(state.lastResult));
  $("#copy-seed").addEventListener("click", () => navigator.clipboard.writeText(String(state.lastResult.seed)));

  $$("#filter-mode button, #filter-backend button").forEach((b) => b.addEventListener("click", () => {
    const key = b.parentElement.id === "filter-mode" ? "mode" : "backend";
    state.filters[key] = b.dataset.v;
    $$(`#${b.parentElement.id} button`).forEach((x) => x.classList.toggle("active", x === b));
    renderGrid();
  }));
  $("#d-close").addEventListener("click", () => $("#detail").close());
  $("#d-reuse").addEventListener("click", () => { $("#detail").close(); reuseSettings(state.detail); });
  $("#d-edit").addEventListener("click", () => { $("#detail").close(); editThis(state.detail); });
  $("#d-delete").addEventListener("click", deleteDetail);
}

async function init() {
  wire();
  state.options = await api("/api/options");
  setMode("generate");
  await refreshHealth();
  setInterval(refreshHealth, 3000);
}

init().catch((err) => showError(`Failed to start: ${err.message}`));
