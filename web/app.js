// neuro-lens : éditeur à gauche, ce que fait chaque couche à droite.

const DEBOUNCE_MS = 450;

// Une phrase par type de couche, pour comprendre ce qu'on voit.
const EXPLAIN = {
  Conv2d: (p) => `Fait glisser ${p.out_channels} filtres ${fmtK(p.kernel_size)} sur l'image : chaque filtre produit une carte qui s'allume là où il reconnaît son motif. Le petit carré sous chaque carte est le filtre lui-même (bleu = poids positif, rouge = négatif).`,
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
  Embedding: (p) => `Remplace chaque token par un vecteur de ${p.embedding_dim} nombres, lu dans une table de ${p.num_embeddings} lignes. Deux fois le même mot donnent deux fois le même vecteur.`,
  MultiheadAttention: (p) => `Chaque token regarde tous les autres et choisit à qui prêter attention, ${p.num_heads} fois en parallèle (${p.num_heads} têtes). Une case claire : ce token (ligne) regarde beaucoup celui-là (colonne).`,
  LayerNorm: () => "Recentre et met à l'échelle le vecteur de chaque token pour stabiliser l'apprentissage.",
};

const FAMILY = {
  Conv2d: "conv", Conv1d: "conv", ConvTranspose2d: "conv",
  MaxPool2d: "pool", AvgPool2d: "pool", AdaptiveAvgPool2d: "pool",
  ReLU: "act", LeakyReLU: "act", Sigmoid: "act", Tanh: "act", GELU: "act", Softmax: "act",
  Linear: "dense", Embedding: "embed", MultiheadAttention: "attn",
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
let off = [];          // [couche, unité] éteintes à la main dans la vue réseau (« et si ? »)
let lastShape = "";    // forme du dernier réseau affiché
let lastResult = null; // pour rejouer la construction
let otherKind = false; // le dernier affichage réussi n'était pas un réseau PyTorch
let lastInput = null; // aperçu de l'entrée, pour montrer son découpage en patchs
let tokens = null;   // mots de la phrase d'entrée, quand le modèle lit du texte
let jumpToOps = false; // l'exemple « MLOps » vient d'être choisi : aller droit à la chaîne
let mlOverride = null; // réglage scikit-learn choisi au curseur, à la place de celui du code
let shownEpoch = null; // null = les derniers poids appris ; sinon le réseau après N époques

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
      ta.addEventListener("input", codeChanged);
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
      editor.onDidChangeModelContent(codeChanged);
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
  document.querySelectorAll("#layers .card, #layers .agent-node").forEach((c) => {
    const on = Number(c.dataset.line) === line;
    c.classList.toggle("active", on);
    if (on && !match) match = c;
  });
  document.querySelectorAll("#graph .graph-col, #blocks .blocks-node").forEach((g) => {
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

// Le code a changé : le réglage fait au curseur ne vaut plus.
function codeChanged() {
  if (window.projectMode) return; // un fichier du projet est ouvert en lecture : pas d'analyse
  mlOverride = null;
  schedule();
}

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

// live = true : le code a le droit d'appeler un vrai modèle de langage (payant), seulement sur clic.
async function run(live = false) {
  const id = ++requestId;
  if (live) setStatus("Claude réfléchit…", "");
  let res;
  try {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: getCode(), pixels, text: $("text").value || null, off, live, epoch: shownEpoch, override: mlOverride }),
    });
    res = await r.json();
  } catch (e) {
    res = { ok: false, error: "Le serveur ne répond pas. Est-il lancé ?", line: null };
  }
  if (id !== requestId) return; // une analyse plus récente est en route
  render(res);
}

// Un modèle scikit-learn ou un agent : pas de couches, donc pas d'entrée à dessiner ni de vue réseau.
function renderOther(res) {
  defiUpdate(null); // pas de défi pour un modèle scikit-learn ou un agent (pour l'instant)
  lastLayers = [];
  lastShape = "";
  tokens = null;
  ["input-card", "graph-card", "blocks-card", "depth-card", "train-card", "dream-card", "attack-card", "landscape-card"].forEach((id) => { $(id).hidden = true; });
  $("prediction").innerHTML = "";
  $("to-ops").hidden = res.kind !== "ml";
  if (res.kind === "ml") {
    setStatus(`${res.estimator} · ${res.n_samples} exemples`, "ok");
    renderML($("layers"), res, highlightLine, revealLine, (name, value) => { mlOverride = { [name]: value }; run(); });
    renderMLOps($("layers"), res, { code: getCode, override: () => mlOverride });
    if (jumpToOps) { jumpToOps = false; $("mlops")?.scrollIntoView({ block: "start" }); }
  } else {
    setStatus(`agent · ${res.steps.length} passage${res.steps.length > 1 ? "s" : ""}`, res.stopped ? "err" : "ok");
    renderAgent($("layers"), res, highlightLine, revealLine, () => run(true));
    renderLLMOps($("layers"), res, highlightLine, revealLine);
  }
}

function render(res) {
  const err = $("error");
  if (res.ok) projectRefresh(res.workspace);
  if (res.ok && typeof renderChain === "function") renderChain(res);
  if (res.ok && res.kind) {
    err.hidden = true;
    setErrorMarker(null);
    otherKind = true;
    renderOther(res);
  } else if (res.ok) {
    otherKind = false;
    $("to-ops").hidden = true;
    $("input-card").hidden = false;
    err.hidden = true;
    setErrorMarker(null);
    setStatus(`${res.layers.length} couches · ${fmtN(res.n_params)} paramètres`, "ok");
    lastLayers = res.layers;
    renderInput(res.input);
    // Le réseau a changé de forme (pas seulement le dessin) : on le reconstruit couche par couche.
    const shape = res.layers.map((l) => `${l.type}:${l.shape.join("x")}`).join("|");
    const build = shape !== lastShape;
    if (build && lastShape && off.length) { off = []; lastShape = shape; return run(); } // autre réseau : on rallume tout
    lastShape = shape;
    lastResult = res;
    renderLayers(res.layers, null, build);
    renderNetwork(res, build);
    renderPrediction(res.prediction, res.training);
    renderTraining(res);
    renderExplore(res);
    renderArcs(res);
    defiUpdate(res);
  } else {
    err.hidden = false;
    err.textContent = (res.line ? `Ligne ${res.line} : ` : "") + res.error;
    setErrorMarker(res.line, res.error);
    setStatus("erreur", "err");
    // On garde les couches qui ont fonctionné, puis une carte rouge là où ça casse.
    if (otherKind && !res.layers) return; // on garde le dernier affichage pendant la frappe
    const partial = res.layers ?? lastLayers;
    renderLayers(partial, res.layers ? res : null);
    $("prediction").innerHTML = "";
  }
}

// La carte d'entrée s'adapte au modèle : zone de dessin pour une image, phrase pour du texte.
function renderInput(input) {
  const isText = input.preview.kind === "text";
  lastInput = input;
  tokens = isText ? input.tokens : null;
  $("text-input").hidden = !isText;
  $("pad-hint").hidden = $("pad-row").hidden = isText;
  $("input-shape").textContent = isText ? `${input.tokens.length} tokens` : fmtShape(input.shape);
  $("saliency-box").hidden = !input.saliency;
  if (input.saliency) paintSaliency(input.preview.maps[0], input.saliency);
  if (!isText) return;
  if (!$("text").value) $("text").value = input.text;
  $("tokens").innerHTML = "";
  input.tokens.forEach((t, i) => {
    const chip = document.createElement("span");
    chip.innerHTML = `<b></b><i>n° ${input.preview.ids[i]}</i>`;
    chip.querySelector("b").textContent = t;
    $("tokens").appendChild(chip);
  });
}

// Éteint ou rallume une unité, puis recalcule tout ce qui en dépend.
function toggleUnit(layer, unit) {
  const at = off.findIndex(([l, u]) => l === layer && u === unit);
  if (at >= 0) off.splice(at, 1); else off.push([layer, unit]);
  run();
}

// Le dessin en gris, et par-dessus en orange les pixels qui pèsent le plus dans la réponse.
function paintSaliency(image, saliency) {
  const cv = $("saliency");
  cv.width = saliency[0].length;
  cv.height = saliency.length;
  const ctx = cv.getContext("2d");
  const data = ctx.createImageData(cv.width, cv.height);
  saliency.flat().forEach((s, i) => {
    const base = (image.flat()[i] ?? 0) * 70;
    const hot = Math.pow(s, 0.6);
    data.data.set([base + hot * (255 - base), base + hot * (150 - base), base * (1 - hot), 255], i * 4);
  });
  ctx.putImageData(data, 0, 0);
}

// Arcs d'attention entre les mots de la phrase : survoler un mot montre à qui il prête attention.
function renderArcs(res) {
  const svg = $("arcs");
  const attn = res.layers.find((l) => l.preview && l.preview.attention);
  const chips = [...$("tokens").children];
  const show = Boolean(attn && tokens && chips.length > 1);
  $("arcs-hint").hidden = !show;
  svg.toggleAttribute("hidden", !show); // un <svg> n'a pas la propriété .hidden des éléments HTML
  if (!show) return;
  const heads = attn.preview.attention;
  const n = Math.min(chips.length, heads[0].length);
  const mean = (i, j) => heads.reduce((acc, h) => acc + (h[i]?.[j] ?? 0), 0) / heads.length;
  const box = $("tokens").getBoundingClientRect();
  const W = box.width, H = 70;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.height = `${H}px`;
  const cx = (k) => { const r = chips[k].getBoundingClientRect(); return r.left - box.left + r.width / 2; };
  const draw = (i) => {
    let html = "";
    for (let j = 0; j < n; j++) {
      const w = mean(i, j);
      if (w < 0.03) continue;
      const a = cx(i), b = cx(j), mid = (a + b) / 2, lift = Math.min(H - 6, Math.abs(b - a) * 0.5 + 10);
      html += `<path d="M${a},4 Q${mid},${4 + lift} ${b},4" stroke-width="${(0.5 + 5 * w).toFixed(1)}" opacity="${(0.25 + 0.75 * w).toFixed(2)}"/>`;
      html += `<text x="${b}" y="${H - 4}" text-anchor="middle">${Math.round(w * 100)} %</text>`;
    }
    svg.innerHTML = html;
  };
  chips.forEach((c, i) => {
    c.onmouseenter = () => { chips.forEach((o) => o.classList.remove("hover")); c.classList.add("hover"); draw(i); };
  });
  $("tokens").onmouseleave = () => { chips.forEach((o) => o.classList.remove("hover")); svg.innerHTML = ""; };
  draw(0);
  chips[0].classList.add("hover");
}

function renderNetwork(res, build) {
  const cols = columnsFromLayers(res.input, res.layers, res.prediction);
  $("relight").hidden = !off.length;
  $("relight").textContent = `Tout rallumer (${off.length})`;
  $("graph-card").hidden = false;
  $("graph-shape").textContent = `${cols.length} colonnes`;
  renderGraph($("graph"), cols, highlightLine, revealLine, build, toggleUnit);
  $("blocks-card").hidden = false;
  const skips = renderBlocks($("blocks"), res, highlightLine, revealLine);
  $("blocks-shape").textContent = skips ? `${skips} raccourci${skips > 1 ? "s" : ""}` : "en ligne droite";
  renderDepth(res, highlightLine, revealLine, build);
}

function revealLine(line) {
  if (!line || !monacoRef) return;
  editor.revealLineInCenter(line);
  editor.setPosition({ lineNumber: line, column: 1 });
  editor.focus();
}

function renderLayers(layers, failure, build) {
  const root = $("layers");
  root.innerHTML = "";
  layers.forEach((l, i) => {
    const card = root.appendChild(layerCard(l));
    if (build) { card.classList.add("build"); card.style.setProperty("--d", `${(i * 0.32).toFixed(2)}s`); }
  });
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
      ${l.group ? `<span class="line-ref">dans ${l.group.type}</span>` : ""}
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
  if (p.kind === "tokens") {
    if (p.attention) wrap.appendChild(attentionEl(p));
    wrap.appendChild(tokenTable(p));
    if (p.points) wrap.appendChild(scatterEl(p.points));
  } else if (p.kind === "maps") {
    if (p.patches) {
      const outer = document.createElement("div");
      outer.appendChild(patchesEl(p.patches));
      const inner = previewEl({ ...p, patches: null });
      outer.appendChild(inner);
      return outer;
    }
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
      if (p.kernels?.[i]) fig.appendChild(kernelEl(p.kernels[i]));
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

// Le filtre lui-même : ses poids, bleu si positif, rouge si négatif.
function kernelEl(kernel) {
  const cv = document.createElement("canvas");
  cv.className = "kernel";
  cv.width = kernel[0].length;
  cv.height = kernel.length;
  cv.title = "Le filtre : bleu = poids positif, rouge = poids négatif";
  const ctx = cv.getContext("2d");
  const data = ctx.createImageData(cv.width, cv.height);
  kernel.flat().forEach((v, j) => {
    const a = Math.abs(v);
    data.data.set(v >= 0 ? [28 + 63 * a, 33 + 107 * a, 48 + 207 * a, 255] : [28 + 227 * a, 33 + 74 * a, 48 + 59 * a, 255], j * 4);
  });
  ctx.putImageData(data, 0, 0);
  return cv;
}

const tokenLabel = (i) => (tokens && tokens[i] !== undefined ? tokens[i] : `${i}`);

// L'image d'entrée découpée en patchs : chaque morceau deviendra un token.
function patchesEl({ size, grid }) {
  const box = document.createElement("div");
  box.className = "patches-box";
  const img = lastInput?.preview.maps?.[0];
  if (!img) return box;
  const scale = img.length / (grid[0] * size[0]); // l'aperçu peut être plus petit que l'image
  const [ph, pw] = [Math.round(size[0] * scale), Math.round(size[1] * scale)];
  const g = document.createElement("div");
  g.className = "patches";
  g.style.gridTemplateColumns = `repeat(${grid[1]}, auto)`;
  for (let r = 0; r < grid[0]; r++) for (let c = 0; c < grid[1]; c++) {
    const cv = document.createElement("canvas");
    cv.width = pw; cv.height = ph;
    const ctx = cv.getContext("2d");
    const data = ctx.createImageData(pw, ph);
    for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
      const v = Math.round((img[r * ph + y]?.[c * pw + x] ?? 0) * 255);
      data.data.set([v, v, v, 255], (y * pw + x) * 4);
    }
    ctx.putImageData(data, 0, 0);
    cv.title = `patch ${r * grid[1] + c}`;
    g.appendChild(cv);
  }
  box.appendChild(g);
  const note = document.createElement("p");
  note.className = "meta";
  note.textContent = `L'image est découpée en ${grid[0] * grid[1]} patchs de ${size[0]}×${size[1]} pixels. ` +
    "Chaque patch devient un token, comme un mot dans une phrase ; les cartes ci-dessous montrent ce que chaque filtre en retient.";
  box.appendChild(note);
  return box;
}

// Un vecteur par token : une ligne de cases, bleue si la valeur est positive, rouge si négative.
function tokenTable(p) {
  const max = Math.max(Math.abs(p.min), Math.abs(p.max)) || 1;
  const table = document.createElement("table");
  table.className = "token-table";
  p.rows.forEach((row, i) => {
    const tr = table.insertRow();
    tr.insertCell().textContent = tokenLabel(i);
    row.forEach((v) => {
      const td = tr.insertCell();
      td.textContent = p.probs ? `${Math.round(v * 100)} %` : v.toFixed(2);
      const color = p.probs ? "76,217,138" : v >= 0 ? "91,140,255" : "255,107,107";
      td.style.background = `rgba(${color},${(Math.abs(v) / max * 0.85).toFixed(2)})`;
      if (p.probs && v === Math.max(...row)) td.className = "best";
    });
    if (p.dim > row.length) tr.insertCell().textContent = "…";
  });
  const box = document.createElement("div");
  box.className = "token-box";
  box.appendChild(table);
  const note = document.createElement("div");
  note.className = "meta";
  note.textContent = (p.probs ? "Chaque ligne se partage 100 % ; la case encadrée est le choix le plus fort"
    : `${Math.min(p.dim, p.rows[0].length)} premières dimensions sur ${p.dim}`) +
    (p.length > p.rows.length ? ` · ${p.rows.length} premiers tokens sur ${p.length}` : "");
  box.appendChild(note);
  return box;
}

// Une grille par tête d'attention : ligne = token qui regarde, colonne = token regardé.
function attentionEl(p) {
  const wrap = document.createElement("div");
  wrap.className = "attn";
  p.attention.forEach((head, h) => {
    const n = head.length;
    const fig = document.createElement("figure");
    const grid = document.createElement("div");
    grid.className = "attn-grid";
    grid.style.gridTemplateColumns = `auto repeat(${n}, 1fr)`;
    head.forEach((row, i) => {
      const label = document.createElement("span");
      label.textContent = tokenLabel(i);
      grid.appendChild(label);
      row.forEach((v, j) => {
        const cell = document.createElement("i");
        cell.style.background = `rgb(${heat(v).slice(0, 3).join(",")})`;
        cell.title = `« ${tokenLabel(i)} » regarde « ${tokenLabel(j)} » : ${(v * 100).toFixed(0)} %`;
        grid.appendChild(cell);
      });
    });
    fig.appendChild(grid);
    const cap = document.createElement("figcaption");
    cap.textContent = `tête ${h}`;
    fig.appendChild(cap);
    wrap.appendChild(fig);
  });
  if (p.heads > p.attention.length) {
    const more = document.createElement("span");
    more.className = "meta";
    more.textContent = `+ ${p.heads - p.attention.length} autres têtes`;
    wrap.appendChild(more);
  }
  return wrap;
}

// Les vecteurs des tokens ramenés à 2 dimensions : deux points proches = deux vecteurs proches.
function scatterEl(points) {
  const size = 220, pad = 26;
  const pos = (v) => pad + ((v + 1) / 2) * (size - 2 * pad);
  const dots = points.map(([x, y], i) => `
    <circle cx="${pos(x)}" cy="${pos(-y)}" r="4"></circle>
    <text x="${pos(x) + 7}" y="${pos(-y) + 4}"></text>`).join("");
  const box = document.createElement("div");
  box.className = "scatter";
  box.innerHTML = `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">${dots}</svg>
    <p class="meta">Les mêmes vecteurs, ramenés de leurs dimensions d'origine à 2 (ACP).
    Deux points proches = deux vecteurs proches.</p>`;
  box.querySelectorAll("text").forEach((t, i) => { t.textContent = tokenLabel(i); });
  return box;
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

function renderPrediction(pred, training) {
  const root = $("prediction");
  if (!pred) { root.innerHTML = ""; return; }
  const bars = pred.probs.map((p, i) => `
    <div class="${i === pred.top ? "top" : ""}" title="${(p * 100).toFixed(1)} %">
      <i style="height:${Math.max(2, p * 100)}%"></i>${i}
    </div>`).join("");
  root.innerHTML = `
    <div class="card" data-family="dense">
      <div class="card-head"><span class="tag">Prédiction</span><span class="shape">classe ${pred.top}</span></div>
      <p class="explain">Probabilité de chaque classe (softmax des scores). ${training
        ? `Le modèle a appris pendant ${training.epochs.length} époque${training.epochs.length > 1 ? "s" : ""} : la réponse est maintenant la sienne.`
        : "Le modèle n'est pas entraîné : ses poids sont aléatoires, donc la réponse l'est aussi. Clique sur « Entraîner »."}</p>
      <div class="probs">${bars}</div>
    </div>`;
}

// Un lien qui contient le code : base64 compatible avec les accents.
const toBase64 = (s) => btoa(String.fromCharCode(...new TextEncoder().encode(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const fromBase64 = (b) => new TextDecoder().decode(Uint8Array.from(atob(b.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0)));

// ---------- Démarrage ----------

(async function main() {
  const select = $("examples");
  // Le menu est rangé par famille, pour qu'on trouve « MLOps » ou « Agents » sans chercher.
  const families = [
    ["MLOps : jusqu'en production", (n) => n.startsWith("MLOps")],
    ["ML classique (scikit-learn)", (n) => n.startsWith("ML classique")],
    ["Agents, LLMOps, AgentOps", (n) => n.startsWith("Agent") || n.startsWith("LLMOps")],
    ["Réseaux de neurones (PyTorch)", () => true],
  ];
  const groups = new Map(families.map(([label]) => [label, Object.assign(document.createElement("optgroup"), { label })]));
  Object.keys(window.NEURO_EXAMPLES).forEach((name) => {
    groups.get(families.find(([, test]) => test(name))[0]).appendChild(new Option(name, name));
  });
  ["Réseaux de neurones (PyTorch)", "ML classique (scikit-learn)", "MLOps : jusqu'en production", "Agents, LLMOps, AgentOps"]
    .forEach((label) => select.appendChild(groups.get(label)));
  select.value = Object.keys(window.NEURO_EXAMPLES)[0];
  $("relight").addEventListener("click", () => { off = []; run(); });
  $("to-ops").addEventListener("click", () => $("mlops")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  select.addEventListener("change", () => { off = []; mlOverride = null; jumpToOps = select.value.startsWith("MLOps"); $("text").value = ""; setCode(window.NEURO_EXAMPLES[select.value]); schedule(); });
  paintPad(seven());
  $("text").addEventListener("input", schedule);
  $("share").addEventListener("click", async () => {
    const url = `${location.origin}${location.pathname}#code=${toBase64(getCode())}`;
    history.replaceState(null, "", url);
    try { await navigator.clipboard.writeText(url); setStatus("lien copié", "ok"); }
    catch (e) { setStatus("lien dans la barre d'adresse", "ok"); }
  });
  $("replay").addEventListener("click", () => {
    if (!lastResult) return;
    renderLayers(lastResult.layers, null, true);
    renderNetwork(lastResult, true);
  });
  const shared = new URLSearchParams(location.hash.slice(1)).get("code");
  await initEditor(shared ? fromBase64(shared) : window.NEURO_EXAMPLES[select.value]);
  run();
})();
