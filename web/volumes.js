// Volumes 3D : ce qui sort de chaque couche, dessiné comme un bloc C × H × W dans l'ordre du calcul.
// Largeur et hauteur = taille de l'image à cette étape, profondeur = nombre de canaux. Les faces sont
// peintes avec les vraies valeurs calculées sur le dessin (face avant : moyenne des canaux ; dessus et
// côté : moyenne le long d'un axe). Un vecteur (après flatten, Linear) devient une colonne. Canvas 2D
// avec une petite perspective, comme la vue 3D des tokens : rien à charger.

const volView = { items: [], yaw: -0.38, pitch: 0.3, hover: -1, hits: [], spin: null, onHover: null, onPick: null };
const VOL_FAMILY_COLOR = { conv: [91, 140, 255], pool: [47, 191, 165], act: [242, 184, 75], dense: [176, 131, 240], input: [139, 147, 167] };
const VOL_CELLS = 12;   // finesse maximale d'une face (cases par côté)
const VOL_MAX = 14;     // au-delà, on échantillonne les couches

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

function volNormalize(grid) {
  let lo = Infinity, hi = -Infinity;
  grid.forEach((r) => r.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  return grid.map((r) => r.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5)));
}

// Un volume à partir d'une couche (ou de l'entrée) : géométrie + les trois textures.
function volItem(name, type, line, shape, preview) {
  if (!preview) return null;
  const fam = volFamily(type);
  if (preview.kind === "maps" && preview.maps && preview.maps.length) {
    const maps = preview.maps;
    const C = preview.channels || shape[1], H = shape[2], W = shape[3];
    const h = maps[0].length, w = maps[0][0].length;
    const front = maps[0].map((row, y) => row.map((_, x) => maps.reduce((s, m) => s + m[y][x], 0) / maps.length));
    const top = maps.map((m) => m[0].map((_, x) => m.reduce((s, row) => s + row[x], 0) / h));
    const side = maps.map((m) => m.map((row) => row.reduce((s, v) => s + v, 0) / w));
    return {
      name, type, line, fam, C, H, W, kind: "maps",
      w: W / 28, h: H / 28, d: Math.min(1.3, 0.13 * Math.sqrt(C)),
      front: volNormalize(volShrink(front, VOL_CELLS)),
      top: volNormalize(volShrink(top, VOL_CELLS)),
      side: volNormalize(volShrink(side, VOL_CELLS)),
    };
  }
  if (preview.kind === "vector" && preview.values && preview.values.length) {
    const L = preview.length || preview.values.length;
    const column = volShrink(preview.values.map((v) => [Math.abs(v)]), 24);
    const front = volNormalize(column);
    return {
      name, type, line, fam, C: L, H: 1, W: 1, kind: "vector",
      w: 0.09, h: Math.min(1.0, 0.045 * Math.sqrt(L)), d: 0.09,
      front, top: [[front[0][0]]], side: front,
    };
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
  // Placement le long de l'axe du calcul, de gauche à droite.
  let x = 0;
  const gap = 0.3;
  items.forEach((it) => { it.x0 = x; x += it.w + gap; });
  volView.total = x - gap;
  volView.items = items;
  volView.onHover = onHover;
  volView.onPick = onPick;
  volView.hover = -1;
  const convs = items.filter((it) => it.fam === "conv").length;
  document.getElementById("volumes-shape").textContent =
    `${items.length} volumes · ${convs} convolution${convs > 1 ? "s" : ""}`;
  if (build) volStartSpin(); else drawVolumes();
}

function volStartSpin() {
  volStopSpin();
  const t0 = performance.now();
  const yaw0 = -0.8;
  const tick = (t) => {
    const k = Math.min(1, (t - t0) / 1500);
    volView.yaw = yaw0 + (1 - (1 - k) ** 3) * 0.42;
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
  const f = 9 / (9 + z2); // perspective douce : le bout de la chaîne reste lisible
  return [cw / 2 + x1 * f * scale, ch / 2 - 18 + y2 * f * scale, z2];
}

function volColor(fam, v, shade, alpha) {
  const [r, g, b] = VOL_FAMILY_COLOR[fam] || VOL_FAMILY_COLOR.input;
  const k = (0.18 + 0.82 * v) * shade;
  return `rgba(${Math.round(22 + (r - 22) * k)},${Math.round(26 + (g - 26) * k)},${Math.round(38 + (b - 38) * k)},${alpha})`;
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
  const scale = Math.min(cw / (volView.total + 1.5), ch / 2.4);
  const P = (p) => volProject([p[0] - volView.total / 2, p[1], p[2]], cw, ch, scale);

  // Les six faces d'un bloc, chacune avec sa texture, son ombrage, et ses axes (u, v) dans la grille.
  const faces = (it) => {
    const x0 = it.x0, x1 = it.x0 + it.w, y0 = -it.h / 2, y1 = it.h / 2, z0 = -it.d / 2, z1 = it.d / 2;
    return [
      { tex: it.front, shade: 0.95, corners: [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]] },  // avant
      { tex: it.front, shade: 0.55, corners: [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]] },  // arrière
      { tex: it.top, shade: 1.0, corners: [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]] },     // dessus
      { tex: it.top, shade: 0.45, corners: [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]] },    // dessous
      { tex: it.side, shade: 0.7, corners: [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]] },    // droite
      { tex: it.side, shade: 0.7, corners: [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]] },    // gauche
    ];
  };

  // Flèches du calcul, derrière les blocs.
  ctx.strokeStyle = "rgba(139,147,167,0.45)";
  ctx.lineWidth = 1.2;
  for (let i = 0; i + 1 < items.length; i++) {
    const a = P([items[i].x0 + items[i].w, 0, 0]), b = P([items[i + 1].x0, 0, 0]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    ctx.beginPath();
    ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - 6 * Math.cos(ang - 0.4), b[1] - 6 * Math.sin(ang - 0.4));
    ctx.lineTo(b[0] - 6 * Math.cos(ang + 0.4), b[1] - 6 * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fillStyle = "rgba(139,147,167,0.6)";
    ctx.fill();
  }

  // Du plus loin au plus proche : les blocs, puis dans chaque bloc les faces par profondeur.
  const order = items.map((it, i) => ({ it, i, z: P([it.x0 + it.w / 2, 0, 0])[2] })).sort((p, q) => q.z - p.z);
  volView.hits = [];
  order.forEach(({ it, i }) => {
    const alpha = hover < 0 || hover === i ? 1 : 0.4;
    const fs = faces(it).map((f) => {
      const q = f.corners.map(P);
      return { ...f, q, z: q.reduce((s, p) => s + p[2], 0) / 4 };
    }).sort((a, b) => b.z - a.z);
    const visible = fs.slice(3); // les trois faces les plus proches : celles qu'on voit
    visible.forEach((f) => {
      const rows = f.tex.length, cols = f.tex[0].length;
      const [a, b, c, d] = f.q; // a→b : axe u (colonnes) ; a→d : axe v (lignes)
      const at = (u, v) => [a[0] + (b[0] - a[0]) * u + (d[0] - a[0]) * v, a[1] + (b[1] - a[1]) * u + (d[1] - a[1]) * v];
      for (let r = 0; r < rows; r++) {
        for (let cidx = 0; cidx < cols; cidx++) {
          const u0 = cidx / cols, u1 = (cidx + 1) / cols, v0 = r / rows, v1 = (r + 1) / rows;
          const pts = [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)];
          ctx.fillStyle = volColor(it.fam, f.tex[r][cidx], f.shade, alpha);
          ctx.beginPath();
          pts.forEach(([px, py], k) => (k ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
          ctx.closePath();
          ctx.fill();
        }
      }
      ctx.strokeStyle = hover === i ? "#7aa2ff" : `rgba(230,232,238,${0.35 * alpha})`;
      ctx.lineWidth = hover === i ? 1.8 : 0.8;
      ctx.beginPath();
      f.q.forEach(([px, py], k) => (k ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.closePath();
      ctx.stroke();
    });
    volView.hits.push({ i, polys: visible.map((f) => f.q) });
  });

  // Étiquettes sous chaque bloc : le type et la forme C × H × W.
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  items.forEach((it, i) => {
    const base = P([it.x0 + it.w / 2, it.h / 2 + 0.06, it.d / 2]);
    const dim = it.kind === "maps" ? `${it.C} × ${it.H} × ${it.W}` : `${it.C}`;
    const y = Math.max(base[1], ch * 0.62) + (i % 2 ? 40 : 12); // en quinconce, sous les blocs
    ctx.fillStyle = hover === i ? "#e6e8ee" : `rgba(139,147,167,${hover < 0 ? 1 : 0.5})`;
    ctx.fillText(it.type === "Entrée" ? "Entrée" : it.type, base[0], y);
    ctx.fillStyle = hover === i ? "#e6e8ee" : `rgba(139,147,167,${hover < 0 ? 0.8 : 0.4})`;
    ctx.fillText(dim, base[0], y + 13);
  });
  ctx.textAlign = "start";
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
  for (let k = volView.hits.length - 1; k >= 0; k--) { // le plus proche a été peint en dernier
    const h = volView.hits[k];
    if (h.polys.some((p) => volPointIn(p, x, y))) return h.i;
  }
  return -1;
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
