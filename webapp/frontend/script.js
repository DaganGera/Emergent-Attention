const REDUCE_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const el = (id) => document.getElementById(id);

const state = {
  mode: "cifar100",
  activeSampleId: null,
};

const modeInputs = document.querySelectorAll('input[name="mode"]');
const sampleStrip = el("sample-strip");
const uploader = el("uploader");
const fileInput = el("file-input");
const scope = el("scope");
const scopeIdle = el("scope-idle");
const scopeToggle = el("scope-toggle");
const scopeStatus = el("scope-status");
const statusLight = document.querySelector(".status-light");
const imgSignal = el("img-signal");
const imgHeatmap = el("img-heatmap");
const decisionBody = el("decision-body");
const decisionDomainTag = el("decision-domain-tag");
const reasoningBody = el("reasoning-body");
const disclaimer = el("disclaimer");

function setStatus(text) {
  statusLight.innerHTML = `<span class="dot"></span> ${text}`;
}

function setBusy(isBusy) {
  scope.classList.toggle("busy", isBusy);
  if (isBusy) {
    const sweep = document.createElement("div");
    sweep.className = "sweep";
    sweep.id = "sweep-el";
    scope.appendChild(sweep);
  } else {
    const existing = document.getElementById("sweep-el");
    if (existing) existing.remove();
  }
}

async function loadSamples(mode) {
  sampleStrip.innerHTML = "";
  const res = await fetch(`/api/samples/${mode}`);
  const items = await res.json();
  for (const item of items) {
    const btn = document.createElement("button");
    btn.className = "sample-chip";
    btn.style.backgroundImage = `url('${item.url}')`;
    btn.title = item.true_label;
    btn.dataset.sampleId = item.id;
    const label = document.createElement("span");
    label.className = "chip-label";
    label.textContent = item.true_label;
    btn.appendChild(label);
    btn.addEventListener("click", () => runSample(mode, item.id, btn));
    sampleStrip.appendChild(btn);
  }
}

function clearActiveChips() {
  document.querySelectorAll(".sample-chip.active").forEach((c) => c.classList.remove("active"));
}

async function runSample(mode, sampleId, chipEl) {
  clearActiveChips();
  if (chipEl) chipEl.classList.add("active");
  const form = new FormData();
  form.append("domain", mode);
  form.append("sample_id", sampleId);
  await runPrediction(form);
}

async function runUpload(mode, file) {
  clearActiveChips();
  const form = new FormData();
  form.append("domain", mode);
  form.append("file", file);
  await runPrediction(form);
}

async function runPrediction(form) {
  setBusy(true);
  setStatus("analyzing image…");
  scopeStatus.textContent = "processing";
  reasoningBody.classList.add("placeholder");
  reasoningBody.innerHTML = `<span class="prompt-glyph">&gt;</span>waiting for a decision to explain…`;

  let result;
  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || res.statusText);
    }
    result = await res.json();
  } catch (err) {
    setBusy(false);
    setStatus("error");
    scopeStatus.textContent = "";
    decisionBody.innerHTML = `<div style="color: var(--red); font-family: var(--font-mono); font-size: 0.82rem;">
      Prediction failed: ${escapeHtml(String(err.message || err))}
    </div>`;
    return;
  }

  setBusy(false);
  scopeStatus.textContent = "";
  renderResult(result);
  runReasoning(result);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function renderResult(result) {
  scopeIdle.hidden = true;
  scopeToggle.hidden = false;
  imgSignal.src = `data:image/png;base64,${result.input_image_b64}`;
  imgHeatmap.src = `data:image/png;base64,${result.heatmap_image_b64}`;
  setScopeView("signal");
  imgSignal.classList.add("visible");

  const domainLabel = result.domain === "busbra" ? "ultrasound" : "photo";
  const ensembleNote = result.ensemble_size > 1 ? ` · ${result.ensemble_size}-seed ensemble` : "";
  decisionDomainTag.textContent = domainLabel + ensembleNote;

  const top = result.top_predictions;
  const rows = top
    .map((p, i) => {
      const pct = (p.confidence * 100).toFixed(1);
      return `<li class="meter-row ${i === 0 ? "top" : ""}">
        <span class="label">${escapeHtml(p.label)}</span>
        <span class="meter-track"><span class="meter-fill" style="width:${pct}%"></span></span>
        <span class="pct">${pct}%</span>
      </li>`;
    })
    .join("");

  decisionBody.innerHTML = `
    <div>
      <span class="decision-label">${escapeHtml(result.predicted_label)}</span>
      <span class="decision-confidence">${(result.confidence * 100).toFixed(1)}%</span>
      <div style="clear:both"></div>
    </div>
    <ul class="meter-list">${rows}</ul>
  `;

  // Animate meters in on next frame (they start at width 0 in the markup above
  // only if we set width after paint; simplest robust approach: they're already
  // at final width via inline style, so just let the CSS transition play from 0.
  requestAnimationFrame(() => {
    decisionBody.querySelectorAll(".meter-fill").forEach((elm) => {
      const target = elm.style.width;
      elm.style.width = "0%";
      requestAnimationFrame(() => { elm.style.width = target; });
    });
  });

  setStatus("decision ready");
}

function setScopeView(view) {
  scopeToggle.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  imgSignal.classList.toggle("visible", view === "signal");
  imgHeatmap.classList.toggle("visible", view === "heatmap");
}

scopeToggle.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-view]");
  if (btn) setScopeView(btn.dataset.view);
});

async function runReasoning(result) {
  reasoningBody.classList.remove("placeholder");
  reasoningBody.innerHTML = `<span class="prompt-glyph">&gt;</span><span id="reasoning-text"></span><span class="cursor" id="reasoning-cursor"></span>`;
  const textEl = el("reasoning-text");

  let text;
  try {
    const res = await fetch("/api/reason", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        domain: result.domain,
        predicted_label: result.predicted_label,
        confidence: result.confidence,
        top_predictions: result.top_predictions,
      }),
    });
    const data = await res.json();
    text = data.explanation;
  } catch (err) {
    text = `Reasoning module unreachable: ${err.message || err}`;
  }

  if (REDUCE_MOTION) {
    textEl.textContent = text;
    const cursor = el("reasoning-cursor");
    if (cursor) cursor.remove();
    return;
  }

  let i = 0;
  const step = () => {
    if (i > text.length) {
      const cursor = el("reasoning-cursor");
      if (cursor) cursor.remove();
      return;
    }
    textEl.textContent = text.slice(0, i);
    i += 2;
    setTimeout(step, 12);
  };
  step();
}

function onModeChange(mode) {
  state.mode = mode;
  disclaimer.hidden = mode !== "busbra";
  el("scope-title").textContent = "Input";
  scopeIdle.hidden = false;
  scopeToggle.hidden = true;
  imgSignal.classList.remove("visible");
  imgHeatmap.classList.remove("visible");
  decisionBody.innerHTML = `<div style="color: var(--ink-faint); font-family: var(--font-mono); font-size: 0.82rem;">awaiting a classification…</div>`;
  reasoningBody.classList.add("placeholder");
  reasoningBody.innerHTML = `<span class="prompt-glyph">&gt;</span>waiting for a decision to explain…`;
  setStatus("model loaded, idle");
  loadSamples(mode);
}

modeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) onModeChange(input.value);
  });
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    runUpload(state.mode, fileInput.files[0]);
  }
});

["dragover", "dragenter"].forEach((evt) =>
  uploader.addEventListener(evt, (e) => {
    e.preventDefault();
    uploader.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  uploader.addEventListener(evt, (e) => {
    e.preventDefault();
    uploader.classList.remove("drag-over");
  })
);
uploader.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) runUpload(state.mode, file);
});

onModeChange("cifar100");
