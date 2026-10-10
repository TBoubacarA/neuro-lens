// neuro-lens : vue « blocs » — une boîte par couche, des flèches pour le trajet réel des données.
//
// Le serveur indique pour chaque couche d'où viennent ses valeurs (`inputs`). On en tire un
// schéma en étages : les couches qui travaillent en parallèle sont côte à côte, et un raccourci
// (connexion résiduelle) contourne les couches qu'il saute. ⊕ marque l'endroit où des chemins se rejoignent.

const BLOCKS = { w: 124, h: 36, gapX: 14, gapY: 36, pad: 26, lane: 12 };

function renderBlocks(root, res, onHover, onPick) {
  root.innerHTML = "";
  const b = BLOCKS;
  const nodes = [{ id: -1, label: "Entrée", sub: res.input.tokens ? `${res.input.tokens.length} tokens` : fmtShape(res.input.shape), family: "input", inputs: [] }];
  res.layers.forEach((l, i) => nodes.push({
    id: i, label: l.type, sub: fmtShape(l.shape), family: FAMILY[l.type] ?? "other",
    inputs: l.inputs ?? (i ? [i - 1] : [-1]), line: l.line, group: l.group, name: l.name,
  }));
  nodes.push({ id: res.layers.length, label: "Sortie", sub: res.prediction ? `classe ${res.prediction.top}` : "", family: "output", inputs: res.output_from ?? [] });
  const byId = new Map(nodes.map((n) => [n.id, n]));

  // Étage d'une boîte : un de plus que la plus basse de ses sources.
  nodes.forEach((n) => { n.rank = n.inputs.length ? Math.max(...n.inputs.map((s) => byId.get(s).rank)) + 1 : 0; });
  const ranks = [];
  nodes.forEach((n) => (ranks[n.rank] ??= []).push(n));
  // Chaque boîte se place sous ses sources ; sur un même étage, on écarte celles qui se chevauchent.
  const step = b.w + b.gapX;
  ranks.forEach((row, r) => {
    row.forEach((n, k) => {
      n.want = n.inputs.length ? n.inputs.reduce((sum, s) => sum + byId.get(s).x, 0) / n.inputs.length : k * step;
    });
    row.sort((p, q) => p.want - q.want);
    row.forEach((n, k) => { n.x = k ? Math.max(n.want, row[k - 1].x + step) : n.want; });
    const shift = row.reduce((sum, n) => sum + n.want - n.x, 0) / row.length;
    row.forEach((n) => { n.x += shift; n.y = b.pad + r * (b.h + b.gapY); });
  });
  const left = Math.min(...nodes.map((n) => n.x));
  nodes.forEach((n) => { n.x += b.pad - left; });
  const inner = Math.max(...nodes.map((n) => n.x + b.w)) - b.pad;

  const skips = [];
  nodes.forEach((n) => n.inputs.forEach((s) => { if (n.rank - byId.get(s).rank > 1) skips.push([byId.get(s), n]); }));
  const width = b.pad * 2 + inner + (skips.length ? 20 + skips.length * b.lane : 0);
  const height = b.pad * 2 + ranks.length * b.h + (ranks.length - 1) * b.gapY;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, class: "blocks-svg" });
  svg.appendChild(el("defs", {})).innerHTML =
    '<marker id="nl-barrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">' +
    '<path d="M0,0 L8,4 L0,8 z" class="graph-arrow"/></marker>';

  // Cadre autour des couches d'un même bloc (TransformerEncoderLayer, un expert...).
  const groups = new Map();
  nodes.forEach((n) => { if (n.group) (groups.get(n.group.name) ?? groups.set(n.group.name, { ...n.group, nodes: [] }).get(n.group.name)).nodes.push(n); });
  groups.forEach((g) => {
    const x0 = Math.min(...g.nodes.map((n) => n.x)) - 7, x1 = Math.max(...g.nodes.map((n) => n.x + b.w)) + 7;
    const y0 = Math.min(...g.nodes.map((n) => n.y)) - 15, y1 = Math.max(...g.nodes.map((n) => n.y + b.h)) + 7;
    svg.appendChild(el("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, rx: 8, class: "blocks-group" }));
    svg.appendChild(el("text", { x: x0 + 6, y: y0 + 10, class: "blocks-group-label" }, `${g.name} · ${g.type}`));
  });

  const edges = svg.appendChild(el("g", { class: "blocks-edges" }));
  let lane = 0;
  nodes.forEach((n) => {
    const merge = n.inputs.length > 1;
    const cx = n.x + b.w / 2;
    const ty = merge ? n.y - b.gapY / 2 + 2 : n.y; // là où les flèches arrivent
    n.inputs.forEach((sid) => {
      const s = byId.get(sid);
      const sx = s.x + b.w / 2, sy = s.y + b.h;
      if (n.rank - s.rank === 1) {
        const endY = merge ? ty - 7 : ty - 1;
        edges.appendChild(el("path", { d: `M${sx},${sy} C${sx},${sy + 14} ${cx},${endY - 14} ${cx},${endY}`, "marker-end": "url(#nl-barrow)" }));
      } else { // raccourci : on contourne par la droite
        const lx = b.pad + inner + 14 + lane++ * b.lane;
        const endX = merge ? cx + 8 : n.x + b.w + 1;
        const endY = merge ? ty : n.y + b.h / 2;
        edges.appendChild(el("path", {
          d: `M${s.x + b.w},${s.y + b.h / 2} H${lx - 8} Q${lx},${s.y + b.h / 2} ${lx},${s.y + b.h / 2 + 8} V${endY - 8} Q${lx},${endY} ${lx - 8},${endY} H${endX}`,
          "marker-end": "url(#nl-barrow)", class: "skip",
        }));
      }
    });
    if (merge) {
      edges.appendChild(el("line", { x1: cx, y1: ty + 7, x2: cx, y2: n.y - 1, "marker-end": "url(#nl-barrow)" }));
      edges.appendChild(el("circle", { cx, cy: ty, r: 7, class: "blocks-merge" }));
      edges.appendChild(el("text", { x: cx, y: ty + 4, class: "blocks-merge-sign" }, "+"));
    }
  });

  nodes.forEach((n) => {
    const g = svg.appendChild(el("g", { class: "blocks-node", "data-family": n.family }));
    if (n.line) g.dataset.line = n.line;
    g.appendChild(el("rect", { x: n.x, y: n.y, width: b.w, height: b.h, rx: 6 }));
    g.appendChild(el("text", { x: n.x + b.w / 2, y: n.y + 15, class: "blocks-label" }, n.label));
    g.appendChild(el("text", { x: n.x + b.w / 2, y: n.y + 28, class: "blocks-sub" }, n.sub));
    if (n.name) g.appendChild(el("title", {}, n.name));
    g.addEventListener("mouseenter", () => onHover(n.line));
    g.addEventListener("mouseleave", () => onHover(null));
    g.addEventListener("click", () => onPick(n.line));
  });
  root.appendChild(svg);
  return skips.length;
}
