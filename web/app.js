// neuro-lens : éditeur à gauche, ce que fait chaque couche à droite.

const DEBOUNCE_MS = 450;

// Une phrase par type de couche, pour comprendre ce qu'on voit.
const EXPLAIN = {
  Conv2d: (p) => `Fait glisser ${p.out_channels} filtres ${fmtK(p.kernel_size)} sur l'image : chaque filtre produit une carte qui s'allume là où il reconnaît son motif.`,
  MaxPool2d: (p) => `Garde la valeur la plus forte de chaque carré ${fmtK(p.kernel_size)} : l'image rétrécit mais les détails importants restent.`,
  AvgPool2d: (p) => `Fait la moyenne de chaque carré ${fmtK(p.kernel_size)} : l'image rétrécit en douceur.`,
  ReLU: () => "Remplace les valeurs négatives par 0 : seules les détections positives passent.",
  LeakyReLU: () => "Comme ReLU, mais laisse passer un peu des valeurs négatives.",
  Sigmoid: () => "Écrase chaque valeur entre 0 et 1.",
  Tanh: () => "Écrase chaque valeur entre -1 et 1.",
  Flatten: () => "Met toutes les cartes bout à bout en une seule longue liste de nombres.",
  Linear: (p) => `Chacun des ${p.out_features} neurones combine les ${p.in_features} valeurs reçues avec ses propres poids.`,
  Dropout: (p) => `Pendant l'entraînement, éteint au hasard ${Math.round((p.p ?? 0.5) * 100)} % des neurones (inactif ici, en mode évaluation).`,
  BatchNorm2d: () => "Recentre et met à l'échelle chaque canal pour stabiliser l'apprentissage.",
  Softmax: () => "Transforme les scores en probabilités qui somment à 1.",
};

const FAMILY = {
  Conv2d: "conv", Conv1d: "conv", ConvTranspose2d: "conv",
  MaxPool2d: "pool", AvgPool2d: "pool", AdaptiveAvgPool2d: "pool",
  ReLU: "act", LeakyReLU: "act", Sigmoid: "act", Tanh: "act", GELU: "act", Softmax: "act",
  Linear: "dense",
};

const $ = (id) => document.getElementById(id);
const fmtK = (k) => (Array.isArray(k) ? k.join("×") : `${k}×${k}`);
const fmtShape = (s) => s.slice(1).join(" × ");
const fmtN = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} k` : `${n}`);

let editor = null;   // Monaco, ou un <textarea> de secours
let monacoRef = null;
let decorations = [];
let timer = null;
let requestId = 0;
let lastLayers = [];

// ---------- Éditeur ----------

function getCode() { return monacoRef ? editor.getValue() : editor.value; }
function setCode(v) { monacoRef ? editor.setValue(v) : (editor.value = v); }

function initEditor(initial) {
  return new Promise((resolve) => {
    const fallback = () => {
      const ta = document.createElement("textarea");
      ta.className = "fallback";
      ta.spellcheck = false;
      ta.value = initial;
      $("editor").appendChild(ta);
      ta.addEventListener("input", schedule);
      ta.addEventListener("keyup", () => focusLine(lineOfTextarea(ta)));
      ta.addEventListener("click", () => focusLine(lineOfTextarea(ta)));
      editor = ta;
      resolve();
    };
    if (!window.require) return fallback();
    window.require.config({ paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs" } });
    window.require(["vs/editor/editor.main"], () => {
      monacoRef = window.monaco;
      editor = monacoRef.editor.create($("editor"), {
        value: initial,
        language: "python",
        theme: "vs-dark",
        fontSize: 13,
        minimap: { enabled: false },
        automaticLayout: true,
        scrollBeyondLastLine: false,
      });
      editor.onDidChangeModelContent(schedule);
      editor.onDidChangeCursorPosition((e) => focusLine(e.position.lineNumber));
      resolve();
    }, fallback);
  });
}

function lineOfTextarea(ta) {
  return ta.value.slice(0, ta.selectionStart).split("\n").length;
}

function highlightLine(line) {
  if (!monacoRef) return;
  decorations = editor.deltaDecorations(decorations, line ? [{
    range: new monacoRef.Range(line, 1, line, 1),
    options: { isWholeLine: true, className: "nl-line", overviewRuler: { color: "#7aa2ff", position: 4 } },
  }] : []);
}

function setErrorMarker(line, message) {
  if (!monacoRef) return;
  const model = editor.getModel();
  const markers = line ? [{
    startLineNumber: line, endLineNumber: line, startColumn: 1,
    endColumn: model.getLineMaxColumn(line), message, severity: monacoRef.MarkerSeverity.Error,
  }] : [];
  monacoRef.editor.setModelMarkers(model, "neuro-lens", markers);
}

// La ligne sous le curseur allume la carte de la couche correspondante.
function focusLine(line) {
  let match = null;
  document.querySelectorAll("#layers .card").forEach((c) => {
    const on = Number(c.dataset.line) === line;
    c.classList.toggle("active", on);
    if (on && !match) match = c;
  });
  document.querySelectorAll("#graph .graph-col").forEach((g) => {
    g.classList.toggle("active", Number(g.dataset.line) === line);
  });
  if (match) match.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------- Zone de dessin ----------

const pad = $("pad");
const pctx = pad.getContext("2d");
let pixels = null; // null = le « 7 » par défaut du serveur
let drawing = false;

function seven() {
  const img = Array.from({ length: 28 }, () => Array(28).fill(0));
  for (let r = 6; r < 9; r++) for (let c = 7; c < 21; c++) img[r][c] = 1;
  for (let i = 0; i < 16; i++) {
    const r = 8 + i, c = 20 - Math.floor(i / 2);
    for (let d = -1; d <= 1; d++) img[r][c + d] = 1;
  }
  return img;
}

function paintPad(img) {
  const data = pctx.createImageData(28, 28);
  img.flat().forEach((v, i) => {
    const g = Math.round(v * 255);
    data.data.set([g, g, g, 255], i * 4);
  });
  pctx.putImageData(data, 0, 0);
}

function stroke(e) {
  const rect = pad.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 28;
  const y = ((e.clientY - rect.top) / rect.height) * 28;
  if (!pixels) pixels = Array.from({ length: 28 }, () => Array(28).fill(0));
  for (let r = Math.floor(y) - 1; r <= Math.floor(y) + 1; r++) {
    for (let c = Math.floor(x) - 1; c <= Math.floor(x) + 1; c++) {
      if (r < 0 || c < 0 || r > 27 || c > 27) continue;
      const d = Math.hypot(r + 0.5 - y, c + 0.5 - x);
      pixels[r][c] = Math.min(1, Math.max(pixels[r][c], 1.3 - d * 0.7));
    }
  }
  paintPad(pixels);
  schedule();
}

pad.addEventListener("pointerdown", (e) => { drawing = true; pad.setPointerCapture(e.pointerId); stroke(e); });
pad.addEventListener("pointermove", (e) => drawing && stroke(e));
pad.addEventListener("pointerup", () => { drawing = false; });
$("clear").addEventListener("click", () => {
  pixels = Array.from({ length: 28 }, () => Array(28).fill(0));
  paintPad(pixels);
  schedule();
});
$("seven").addEventListener("click", () => { pixels = null; paintPad(seven()); schedule(); });

// ---------- Analyse ----------

function schedule() {
  clearTimeout(timer);
  setStatus("…", "");
  timer = setTimeout(run, DEBOUNCE_MS);
}

function setStatus(text, cls) {
  const s = $("status");
  s.textContent = text;
  s.className = `status ${cls}`;
}

async function run() {
  const id = ++requestId;
  let res;
  try {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: getCode(), pixels }),
    });
    res = await r.json();
  } catch (e) {
    res = { ok: false, error: "Le serveur ne répond pas. Est-il lancé ?", line: null };
  }
  if (id !== requestId) return; // une analyse plus récente est en route
  render(res);
}

function render(res) {
  const err = $("error");
  if (res.ok) {
    err.hidden = true;
    setErrorMarker(null);
    setStatus(`${res.layers.length} couches · ${fmtN(res.n_params)} paramètres`, "ok");
    lastLayers = res.layers;
    renderLayers(res.layers, null);
    renderNetwork(res);
    renderPrediction(res.prediction);
  } else {
    err.hidden = false;
    err.textContent = (res.line ? `Ligne ${res.line} : ` : "") + res.error;
    setErrorMarker(res.line, res.error);
    setStatus("erreur", "err");
    // On garde les couches qui ont fonctionné, puis une carte rouge là où ça casse.
    const partial = res.layers ?? lastLayers;
    renderLayers(partial, res.layers ? res : null);
    $("prediction").innerHTML = "";
  }
}

function renderNetwork(res) {
  const cols = columnsFromLayers(res.input, res.layers, res.prediction);
  $("graph-card").hidden = false;
  $("graph-shape").textContent = `${cols.length} colonnes`;
  renderGraph($("graph"), cols, highlightLine, revealLine);
}

function revealLine(line) {
  if (!line || !monacoRef) return;
  editor.revealLineInCenter(line);
  editor.setPosition({ lineNumber: line, column: 1 });
  editor.focus();
}

function renderLayers(layers, failure) {
  const root = $("layers");
  root.innerHTML = "";
  layers.forEach((l) => root.appendChild(layerCard(l)));
  if (failure) {
    const c = document.createElement("div");
    c.className = "card failed";
    c.dataset.line = failure.line ?? "";
    c.innerHTML = `<div class="card-head"><span class="tag">✗ Ça casse ici</span>
      ${failure.line ? `<span class="line-ref">ligne ${failure.line}</span>` : ""}</div>
      <p class="explain"></p>`;
    c.querySelector(".explain").textContent = failure.error;
    root.appendChild(c);
  }
}

function layerCard(l) {
  const c = document.createElement("div");
  c.className = "card";
  c.dataset.family = FAMILY[l.type] ?? "other";
  c.dataset.line = l.line ?? "";
  const explain = EXPLAIN[l.type] ? EXPLAIN[l.type](l.params) : "";
  c.innerHTML = `
    <div class="card-head">
      <span class="tag">${l.type}</span>
      <span class="name">${l.name}</span>
      ${l.line ? `<span class="line-ref">ligne ${l.line}</span>` : ""}
      <span class="shape" title="forme de la sortie">${fmtShape(l.shape)}</span>
    </div>
    ${explain ? `<p class="explain">${explain}</p>` : ""}`;
  c.appendChild(previewEl(l.preview));
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `${l.n_params ? fmtN(l.n_params) + " paramètres · " : ""}valeurs de ${l.preview.min} à ${l.preview.max}`;
  c.appendChild(meta);
  c.addEventListener("mouseenter", () => highlightLine(l.line));
  c.addEventListener("mouseleave", () => highlightLine(null));
  c.addEventListener("click", () => revealLine(l.line));
  return c;
}

function previewEl(p) {
  const wrap = document.createElement("div");
  if (p.kind === "maps") {
    wrap.className = "maps";
    p.maps.forEach((m, i) => {
      const fig = document.createElement("figure");
      const cv = document.createElement("canvas");
      cv.width = m[0].length;
      cv.height = m.length;
      const ctx = cv.getContext("2d");
      const data = ctx.createImageData(cv.width, cv.height);
      m.flat().forEach((v, j) => data.data.set(heat(v), j * 4));
      ctx.putImageData(data, 0, 0);
      fig.appendChild(cv);
      const cap = document.createElement("figcaption");
      cap.textContent = `canal ${i}`;
      fig.appendChild(cap);
      wrap.appendChild(fig);
    });
    if (p.channels > p.maps.length) {
      const more = document.createElement("span");
      more.className = "meta";
      more.textContent = `+ ${p.channels - p.maps.length} autres cartes`;
      wrap.appendChild(more);
    }
  } else {
    wrap.className = "vector";
    const max = Math.max(Math.abs(p.min), Math.abs(p.max)) || 1;
    p.values.forEach((v) => {
      const s = document.createElement("span");
      s.style.height = `${Math.max(4, (Math.abs(v) / max) * 100)}%`;
      s.style.background = v >= 0 ? "var(--dense)" : "var(--danger)";
      s.title = v;
      wrap.appendChild(s);
    });
    if (p.length > p.values.length) wrap.title = `${p.values.length} premières valeurs sur ${p.length}`;
  }
  return wrap;
}

// Échelle de couleur sombre -> bleu -> jaune pour les cartes d'activation.
function heat(v) {
  const stops = [[15, 17, 23], [40, 70, 160], [90, 160, 255], [255, 220, 90]];
  const t = v * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const f = t - i;
  const [a, b] = [stops[i], stops[i + 1]];
  return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f)).concat(255);
}

function renderPrediction(pred) {
  const root = $("prediction");
  if (!pred) { root.innerHTML = ""; return; }
  const bars = pred.probs.map((p, i) => `
    <div class="${i === pred.top ? "top" : ""}" title="${(p * 100).toFixed(1)} %">
      <i style="height:${Math.max(2, p * 100)}%"></i>${i}
    </div>`).join("");
  root.innerHTML = `
    <div class="card" data-family="dense">
      <div class="card-head"><span class="tag">Prédiction</span><span class="shape">classe ${pred.top}</span></div>
      <p class="explain">Probabilité de chaque classe (softmax des scores). Le modèle n'est pas entraîné :
        ses poids sont aléatoires, donc la réponse l'est aussi pour l'instant.</p>
      <div class="probs">${bars}</div>
    </div>`;
}

// ---------- Démarrage ----------

(async function main() {
  const select = $("examples");
  Object.keys(window.NEURO_EXAMPLES).forEach((name) => select.add(new Option(name, name)));
  select.addEventListener("change", () => { setCode(window.NEURO_EXAMPLES[select.value]); schedule(); });
  paintPad(seven());
  await initEditor(window.NEURO_EXAMPLES[select.value]);
  run();
})();
