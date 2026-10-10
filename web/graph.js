// neuro-lens : vue « réseau » — des neurones en colonnes, reliés par leurs connexions.
//
// Le dessin part d'une description neutre : une liste de colonnes
//   { label, sub, family, count, values, link, line, top }
// `link` dit comment la colonne se relie à la précédente : "full" (tous vers tous)
// ou "one" (un vers un). Tout ce qui sait produire ces colonnes peut être dessiné ici.

const SVG_NS = "http://www.w3.org/2000/svg";
// Jusqu'à `maxNodes` unités, on les dessine toutes : 8 filtres dans le code = 8 nœuds à l'écran.
const GRAPH = { colGap: 140, rowGap: 37, radius: 15, maxNodes: 16, padX: 46, padTop: 20, padBottom: 46, stagger: 0.32 };

// Couches qui transforment les valeurs sans changer le nombre de neurones :
// on les note sous la colonne précédente au lieu d'en créer une nouvelle.
const INLINE = new Set([
  "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "Softmax", "Dropout", "BatchNorm1d", "BatchNorm2d",
]);

// Nombre d'unités d'une sortie : les canaux pour des cartes, les tokens pour un texte,
// les valeurs (ou les dimensions du vecteur de chaque token) sinon.
function unitCount(shape, preview) {
  return preview.kind === "maps" || preview.kind === "text" ? shape[1] : shape[shape.length - 1];
}

// Une valeur d'activité entre 0 et 1 par unité, quand le serveur l'a renvoyée.
function unitValues(preview) {
  if (preview.kind === "text") return preview.ids.map(() => 1);
  if (preview.kind === "maps") {
    return preview.maps.map((m) => {
      const flat = m.flat();
      return flat.reduce((a, b) => a + b, 0) / flat.length;
    });
  }
  const max = Math.max(Math.abs(preview.min), Math.abs(preview.max)) || 1;
  return preview.values.map((v) => Math.abs(v) / max);
}

function columnsFromLayers(input, layers, prediction) {
  const cols = [{
    label: "Entrée", sub: input.shape.slice(1).join("×"), family: "input",
    count: unitCount(input.shape, input.preview), values: unitValues(input.preview), link: null, line: null,
    maps: input.preview.maps, gray: true,
  }];
  layers.forEach((l) => {
    const prev = cols[cols.length - 1];
    const count = unitCount(l.shape, l.preview);
    if (INLINE.has(l.type) && cols.length > 1 && count === prev.count) {
      prev.sub = `${prev.sub} → ${l.type}`;
      prev.values = unitValues(l.preview);
      prev.maps = l.preview.maps;
      return;
    }
    const family = FAMILY[l.type] ?? "other";
    cols.push({
      label: l.type, sub: l.shape.slice(1).join("×"), family, count,
      values: unitValues(l.preview),
      maps: l.preview.maps,
      link: family === "pool" && count === prev.count ? "one" : "full",
      line: l.line ?? null,
    });
  });
  if (prediction) {
    const last = cols[cols.length - 1];
    last.family = "output";
    last.values = prediction.probs;
    last.top = prediction.top;
  }
  return cols;
}

// Indices des unités dessinées ; `null` marque les points de suspension.
function visibleUnits(count, max) {
  if (count <= max) return Array.from({ length: count }, (_, i) => i);
  const head = Array.from({ length: max - 2 }, (_, i) => i);
  return [...head, null, count - 1];
}

// Petite image d'une carte d'activation, pour la poser à la place du rond.
function mapImage(map, gray) {
  const cv = document.createElement("canvas");
  cv.width = map[0].length;
  cv.height = map.length;
  const ctx = cv.getContext("2d");
  const data = ctx.createImageData(cv.width, cv.height);
  map.flat().forEach((v, j) => data.data.set(gray ? [v * 255, v * 255, v * 255, 255] : heat(v), j * 4));
  ctx.putImageData(data, 0, 0);
  return cv.toDataURL();
}

function el(name, attrs, text) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderGraph(root, cols, onHover, onPick, build) {
  root.innerHTML = "";
  const g = GRAPH;
  const placed = cols.map((c, i) => {
    const units = visibleUnits(c.count, g.maxNodes);
    return { ...c, units, x: g.padX + i * g.colGap };
  });
  const rows = Math.max(...placed.map((c) => c.units.length));
  const bodyH = (rows - 1) * g.rowGap;
  const width = g.padX * 2 + (placed.length - 1) * g.colGap;
  const height = g.padTop + g.radius * 2 + bodyH + g.padBottom;
  const yOf = (c, k) => g.padTop + g.radius + (bodyH - (c.units.length - 1) * g.rowGap) / 2 + k * g.rowGap;

  // `build` : le réseau se construit sous les yeux, une couche après l'autre.
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, class: build ? "graph-svg build" : "graph-svg" });
  const delay = (i, late) => `--d:${(i * g.stagger + (late ? g.stagger / 2 : 0)).toFixed(2)}s`;
  svg.appendChild(el("defs", {})).innerHTML =
    '<marker id="nl-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">' +
    '<path d="M0,0 L8,4 L0,8 z" class="graph-arrow"/></marker>';

  const edges = svg.appendChild(el("g", { class: "graph-edges" }));
  placed.forEach((c, i) => {
    if (!i) return;
    const p = placed[i - 1];
    c.units.forEach((u, k) => {
      if (u === null) return;
      p.units.forEach((v, j) => {
        if (v === null) return;
        if (c.link === "one" && u !== v) return;
        edges.appendChild(el("line", {
          x1: p.x + g.radius, y1: yOf(p, j), x2: c.x - g.radius - 2, y2: yOf(c, k),
          "marker-end": "url(#nl-arrow)", "data-col": i, style: delay(i - 1, true),
        }));
      });
    });
  });

  placed.forEach((c, i) => {
    const group = svg.appendChild(el("g", { class: "graph-col", "data-family": c.family, "data-col": i, style: delay(i) }));
    if (c.line) group.dataset.line = c.line;
    c.units.forEach((u, k) => {
      const y = yOf(c, k);
      if (u === null) {
        group.appendChild(el("text", { x: c.x, y: y + 5, class: "graph-dots" }, "⋮"));
        return;
      }
      const v = c.values[u];
      let node;
      if (c.maps?.[u]) { // un canal : on montre ce qu'il voit vraiment du dessin
        group.appendChild(el("image", {
          href: mapImage(c.maps[u], c.gray), x: c.x - g.radius, y: y - g.radius,
          width: g.radius * 2, height: g.radius * 2, class: "graph-map", preserveAspectRatio: "none",
        }));
        node = el("rect", { x: c.x - g.radius, y: y - g.radius, width: g.radius * 2, height: g.radius * 2, rx: 3, class: "graph-node framed" });
      } else {
        node = el("circle", { cx: c.x, cy: y, r: g.radius, class: "graph-node" });
        // Plus le neurone est actif, plus il est lumineux.
        node.style.fillOpacity = v === undefined ? 0.35 : 0.25 + 0.75 * Math.min(1, v);
      }
      if (c.top === u) node.classList.add("top");
      node.appendChild(el("title", {}, `${c.label} · unité ${u}${v === undefined ? "" : ` · activité ${v.toFixed(2)}`}`));
      group.appendChild(node);
      if (c.family === "output") group.appendChild(el("text", { x: c.x, y: y + 4, class: "graph-idx" }, u));
    });
    const base = g.padTop + g.radius * 2 + bodyH + 18;
    group.appendChild(el("text", { x: c.x, y: base, class: "graph-label" }, c.label));
    group.appendChild(el("text", { x: c.x, y: base + 14, class: "graph-sub" },
      `${c.count} ${c.count > 1 ? "unités" : "unité"}`));
    group.appendChild(el("title", {}, `${c.label} — ${c.sub}`));
    group.addEventListener("mouseenter", () => { markFocus(svg, i); onHover(c.line); });
    group.addEventListener("mouseleave", () => { markFocus(svg, null); onHover(null); });
    group.addEventListener("click", () => onPick(c.line));
  });

  root.appendChild(svg);
}

// Met en avant une colonne et les connexions qui y arrivent.
function markFocus(svg, col) {
  svg.classList.toggle("has-focus", col !== null);
  svg.querySelectorAll("[data-col]").forEach((n) => n.classList.toggle("focus", Number(n.dataset.col) === col));
}
