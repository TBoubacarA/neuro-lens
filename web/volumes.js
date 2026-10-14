// Volumes 3D : ce qui sort de chaque couche d'un réseau à convolutions, dans le même style que la
// vue 3D des tokens : une plaque par étape, empilées dans l'ordre du calcul. La plaque fait la taille
// de l'image à cette étape (28 × 28 puis 14 × 14 puis 7 × 7 : on voit l'image rétrécir), et les canaux
// forment une petite pile de feuilles derrière elle (une feuille par canal, jusqu'à quatre). Orange :
// une zone plus active que la moyenne de la carte, bleu : moins. Après flatten ou Linear, la sortie est
// un vecteur : une colonne de cases. Canvas 2D avec une petite perspective, rien à charger.

const volView = { items: [], yaw: -0.75, pitch: 0.42, hover: -1, hits: [], spin: null, onHover: null, onPick: null };
const VOL_CELLS = 14;   // finesse maximale d'une plaque (cases par côté)
const VOL_MAX = 14;     // au-delà, on échantillonne les couches
const VOL_SHEETS = 4;   // feuilles dessinées par couche (une par canal)
const VOL_FAMILY_COLOR = { conv: [91, 140, 255], pool: [47, 191, 165], act: [242, 184, 75], dense: [176, 131, 240], input: [139, 147, 167] };

function volFamily(type) {
  return (typeof FAMILY === "object" && FAMILY[type]) || (/Pool/.test(type) ? "pool" : /Norm|Flatten|Dropout|Identity/.test(type) ? "input" : "dense");
}

// Réduit une grille à au plus n × n cases (moyenne par bloc), pour garder un dessin fluide.
function volShrink(grid, n) {
  const H = grid.length, W = grid[0].length;
  const h = Math.min(n, H), w = Math.min(n, W);
  const out = Array.from({ length: h }, () => Array(w).fill(0));
  const cnt = Array.from({ length: h }, () => Array(w).fill(0));
  for (let y = 0; y < H; y++) {
    const yy = Math.floor((y * h) / H);
    for (let x = 0; x < W; x++) {
      const xx = Math.floor((x * w) / W);
      out[yy][xx] += grid[y][x];
      cnt[yy][xx] += 1;
    }
  }
  return out.map((row, y) => row.map((v, x) => v / cnt[y][x]));
}

// Centre une grille sur sa moyenne : positif = plus actif que la moyenne, négatif = moins.
function volCenter(grid) {
  let s = 0, n = 0;
  grid.forEach((r) => r.forEach((v) => { s += v; n += 1; }));
  const mean = n ? s / n : 0;
  return grid.map((r) => r.map((v) => v - mean));
}

// Une étape à partir d'une couche (ou de l'entrée) : sa taille et ses feuilles (une par canal).
function volItem(name, type, line, shape, preview) {
  if (!preview) return null;
  const fam = volFamily(type);
  if (preview.kind === "maps" && preview.maps && preview.maps.length) {
    const C = preview.channels || shape[1], H = shape[2], W = shape[3];
    const sheets = preview.maps.slice(0, VOL_SHEETS).map((m) => volCenter(volShrink(m, VOL_CELLS)));
    return { name, type, line, fam, C, H, W, kind: "maps", sheets };
  }
  if (preview.kind === "vector" && preview.values && preview.values.length) {
    const L = preview.length || preview.values.length;
    const column = volShrink(preview.values.map((v) => [Math.abs(v)]), 28);
    return { name, type, line, fam, C: L, H: 1, W: 1, kind: "vector", sheets: [volCenter(column)] };
  }
  return null;
}

function renderVolumes(res, onHover, onPick, build) {
  const card = document.getElementById("volumes-card");
  let items = [];
  if (res.input && res.input.preview && res.input.preview.kind === "maps") {
    items.push(volItem("entrée", "Entrée", null, res.input.shape, res.input.preview));
  }
  res.layers.forEach((l) => { const it = volItem(l.name, l.type, l.line, l.shape, l.preview); if (it) items.push(it); });
  items = items.filter(Boolean);
  const hasMaps = items.some((it) => it.kind === "maps");
  card.hidden = !hasMaps || items.length < 2;
  if (card.hidden) { volStopSpin(); return; }
  if (items.length > VOL_MAX) {
    const step = (items.length - 1) / (VOL_MAX - 1);
    items = Array.from({ length: VOL_MAX }, (_, i) => items[Math.round(i * step)]);
  }
  // La plus grande image de la chaîne donne la taille de référence des plaques.
  const ref = Math.max(...items.filter((it) => it.kind === "maps").map((it) => Math.max(it.H, it.W)));
  items.forEach((it) => {
    if (it.kind === "maps") { it.w = (1.1 * it.W) / ref; it.h = (1.1 * it.H) / ref; }
    else { it.w = 0.08; it.h = Math.min(1.1, 0.06 * Math.sqrt(it.C)); }
  });
  volView.items = items;
  volView.onHover = onHover;
  volView.onPick = onPick;
  volView.hover = -1;
  const convs = items.filter((it) => it.fam === "conv").length;
  document.getElementById("volumes-shape").textContent =
    `${items.length} étapes · ${convs} convolution${convs > 1 ? "s" : ""}`;
  if (build) volStartSpin(); else drawVolumes();
}

function volStartSpin() {
  volStopSpin();
  const t0 = performance.now();
  const yaw0 = -1.25;
  const tick = (t) => {
    const k = Math.min(1, (t - t0) / 1600);
    volView.yaw = yaw0 + (1 - (1 - k) ** 3) * 0.5;
    drawVolumes();
    volView.spin = k < 1 ? requestAnimationFrame(tick) : null;
  };
  volView.spin = requestAnimationFrame(tick);
}

function volStopSpin() {
  if (volView.spin) cancelAnimationFrame(volView.spin);
  volView.spin = null;
}

function volProject(p, cw, ch, scale) {
  const { yaw, pitch } = volView;
  const x1 = p[0] * Math.cos(yaw) + p[2] * Math.sin(yaw);
  const z1 = -p[0] * Math.sin(yaw) + p[2] * Math.cos(yaw);
  const y2 = p[1] * Math.cos(pitch) - z1 * Math.sin(pitch);
  const z2 = p[1] * Math.sin(pitch) + z1 * Math.cos(pitch);
  const f = 4 / (4 + z2);
  return [cw / 2 + x1 * f * scale, ch / 2 + y2 * f * scale, z2];
}

function volColor(v, max, alpha) {
  const a = Math.min(1, Math.abs(v) / (max || 1));
  const [r, g, b] = v >= 0 ? [242, 160, 75] : [91, 140, 255];
  return `rgba(${r},${g},${b},${(0.12 + 0.88 * a) * alpha})`;
}

function volQuad(ctx, q) {
  ctx.beginPath();
  q.forEach(([px, py], k) => (k ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
  ctx.closePath();
}

function drawVolumes() {
  const canvas = document.getElementById("volumes");
  const { items, hover } = volView;
  if (!items.length) return;
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const scale = Math.min(cw / 3.1, ch / 1.9);

  // Repère : x et y = l'image, z = profondeur (ordre des couches). Les feuilles d'une même couche
  // se suivent de près ; les couches sont plus espacées.
  const n = items.length;
  const sheetGap = 0.06;
  const D = Math.min(3.2, Math.max(2.2, 0.3 * (n - 1)));
  const zOf = (i) => D / 2 - (n === 1 ? 0 : (i / (n - 1)) * D);
  const P = (p) => volProject(p, cw, ch, scale);

  // Le sol : un cadre sous la chaîne, pour sentir la profondeur, et l'axe du calcul.
  const floorY = 0.62;
  ctx.strokeStyle = "rgba(139,147,167,0.18)";
  ctx.lineWidth = 1;
  for (let i = 0; i < n; i++) {
    const a = P([-0.7, floorY, zOf(i)]), b = P([0.7, floorY, zOf(i)]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  [-0.7, 0.7].forEach((x) => {
    const a = P([x, floorY, zOf(0)]), b = P([x, floorY, zOf(n - 1)]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  });

  // On peint du plus loin au plus proche : les couches, puis dans chaque couche les feuilles.
  const order = items.map((it, i) => ({ it, i, z: P([0, 0, zOf(i)])[2] })).sort((p, q) => q.z - p.z);
  const toward = Math.cos(volView.yaw) * Math.cos(volView.pitch) >= 0 ? 1 : -1; // de quel côté on regarde
  volView.hits = [];
  order.forEach(({ it, i }) => {
    const alpha = hover < 0 || hover === i ? 1 : 0.35;
    const sheets = it.sheets;
    const zs = sheets.map((_, s) => zOf(i) - toward * s * sheetGap); // la feuille 0 devant, les autres derrière
    let max = 0;
    sheets.forEach((g) => g.forEach((r) => r.forEach((v) => { max = Math.max(max, Math.abs(v)); })));
    const w = it.w, h = it.h;
    for (let s = sheets.length - 1; s >= 0; s--) {
      const z = zs[s], g = sheets[s];
      const rows = g.length, cols = g[0].length;
      const cellW = w / cols, cellH = h / rows;
      const fade = s === 0 ? 1 : 0.55; // les canaux derrière, plus discrets
      g.forEach((r, y) => {
        r.forEach((v, x) => {
          const x0 = -w / 2 + x * cellW, y0 = -h / 2 + y * cellH;
          const q = [[x0, y0], [x0 + cellW, y0], [x0 + cellW, y0 + cellH], [x0, y0 + cellH]].map(([px, py]) => P([px, py, z]));
          ctx.fillStyle = volColor(v, max, alpha * fade);
          volQuad(ctx, q);
          ctx.fill();
        });
      });
      const corners = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]].map(([px, py]) => P([px, py, z]));
      ctx.strokeStyle = s === 0 && hover === i ? "#7aa2ff" : `rgba(139,147,167,${(s === 0 ? 0.5 : 0.25) * alpha})`;
      ctx.lineWidth = s === 0 && hover === i ? 2 : 1;
      volQuad(ctx, corners);
      ctx.stroke();
      if (s === 0) volView.hits.push({ i, corners });
    }
  });

  // Noms des étapes au-dessus des plaques, sur trois rangées en quinconce pour rester lisibles.
  ctx.font = "11px system-ui, sans-serif";
  const rowEnd = [-Infinity, -Infinity, -Infinity, -Infinity];
  items.forEach((it, i) => {
    const dim = it.kind === "maps" ? `${it.C} × ${it.H} × ${it.W}` : `${it.C}`;
    const text = `${i + 1}. ${it.type === "Entrée" ? "entrée" : it.type} · ${dim}`;
    const top = P([it.w / 2 + 0.04, -it.h / 2 - 0.04, zOf(i)]);
    // Quatre rangées en quinconce, et un nom qui chevaucherait son voisin de rangée est poussé à droite.
    const r = i % 4, wText = ctx.measureText(text).width;
    let x = Math.max(top[0], rowEnd[r] + 8);
    x = Math.min(x, cw - wText - 4);
    rowEnd[r] = x + wText;
    const y = 14 + r * 14; // les noms vivent en haut, en quatre rangées ; un fil descend vers la plaque
    ctx.fillStyle = hover === i ? "#e6e8ee" : `rgba(139,147,167,${hover < 0 ? 1 : 0.5})`;
    ctx.fillText(text, x, y);
    if (y < top[1] - 6) { // un petit fil relie le nom à sa plaque quand il a été remonté
      ctx.strokeStyle = `rgba(139,147,167,${hover === i ? 0.8 : 0.3})`;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, y + 3); ctx.lineTo(top[0], top[1]); ctx.stroke();
    }
  });

  // La taille de l'image, à gauche de la première plaque : on la voit rétrécir le long de la chaîne.
  const first = items[0];
  ctx.fillStyle = "#e6e8ee";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right";
  const p0 = P([-first.w / 2 - 0.06, 0, zOf(0)]);
  ctx.fillText(first.kind === "maps" ? `${first.H} × ${first.W}` : `${first.C}`, p0[0], p0[1] + 4);
  ctx.textAlign = "left";
}

function volPointIn(poly, x, y) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function volAt(e) {
  const canvas = document.getElementById("volumes");
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  // hits est rangé du plus loin au plus proche : la dernière plaque touchée est devant.
  const hit = volView.hits.filter((h) => volPointIn(h.corners, x, y)).pop();
  return hit ? hit.i : -1;
}

(function initVolumes() {
  const canvas = document.getElementById("volumes");
  if (!canvas) return;
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    volStopSpin();
    drag = { x: e.clientX, y: e.clientY, yaw: volView.yaw, pitch: volView.pitch, moved: false };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      volView.yaw = drag.yaw + dx * 0.008;
      volView.pitch = Math.max(-1.3, Math.min(1.3, drag.pitch + dy * 0.008));
      drawVolumes();
      return;
    }
    const i = volAt(e);
    if (i !== volView.hover) {
      volView.hover = i;
      drawVolumes();
      if (volView.onHover) volView.onHover(i >= 0 ? volView.items[i].line : null);
    }
  });
  canvas.addEventListener("pointerup", (e) => {
    const click = drag && !drag.moved;
    drag = null;
    if (click) {
      const i = volAt(e);
      if (i >= 0 && volView.onPick) volView.onPick(volView.items[i].line);
    }
  });
  canvas.addEventListener("pointerleave", () => {
    if (drag || volView.hover < 0) return;
    volView.hover = -1;
    drawVolumes();
    if (volView.onHover) volView.onHover(null);
  });
  window.addEventListener("resize", drawVolumes);
})();
