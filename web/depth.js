// Vue 3D façon BERT : la sortie de chaque couche est une plaque (une ligne par mot, une case
// par nombre du vecteur), et les plaques s'empilent dans l'ordre du calcul. Dessin en canvas 2D
// avec une petite projection en perspective : pas de bibliothèque 3D à charger.

const depthView = { slabs: [], tokens: [], yaw: -0.75, pitch: 0.42, hover: -1, spin: null, onHover: null, onPick: null };
const MAX_SLABS = 14;

function renderDepth(res, onHover, onPick, build) {
  const card = document.getElementById("depth-card");
  let slabs = res.layers
    .filter((l) => l.preview && l.preview.kind === "tokens" && l.preview.wide && l.type !== "Dropout")
    .map((l) => ({ name: l.name, type: l.type, line: l.line, rows: l.preview.wide, dim: l.preview.dim }));
  if (slabs.length > MAX_SLABS) {
    const step = (slabs.length - 1) / (MAX_SLABS - 1);
    slabs = Array.from({ length: MAX_SLABS }, (_, i) => slabs[Math.round(i * step)]);
  }
  card.hidden = slabs.length < 2;
  if (card.hidden) { stopSpin(); return; }
  depthView.slabs = slabs;
  depthView.tokens = (res.input && res.input.tokens) || []; // pas de mots (ViT) : on numérote
  depthView.onHover = onHover;
  depthView.onPick = onPick;
  document.getElementById("depth-shape").textContent =
    `${slabs.length} couches · ${slabs[0].rows.length} tokens`;
  if (build) startSpin();
  drawDepth();
}

// Petite rotation au premier affichage, pour qu'on voie tout de suite que c'est en 3D.
function startSpin() {
  stopSpin();
  const t0 = performance.now();
  const yaw0 = -1.25;
  const tick = (t) => {
    const k = Math.min(1, (t - t0) / 1600);
    depthView.yaw = yaw0 + (1 - (1 - k) ** 3) * 0.5;
    drawDepth();
    depthView.spin = k < 1 ? requestAnimationFrame(tick) : null;
  };
  depthView.spin = requestAnimationFrame(tick);
}

function stopSpin() {
  if (depthView.spin) cancelAnimationFrame(depthView.spin);
  depthView.spin = null;
}

function project(p, cw, ch, scale) {
  const { yaw, pitch } = depthView;
  const x1 = p[0] * Math.cos(yaw) + p[2] * Math.sin(yaw);
  const z1 = -p[0] * Math.sin(yaw) + p[2] * Math.cos(yaw);
  const y2 = p[1] * Math.cos(pitch) - z1 * Math.sin(pitch);
  const z2 = p[1] * Math.sin(pitch) + z1 * Math.cos(pitch);
  const f = 4 / (4 + z2);
  return [cw / 2 + x1 * f * scale, ch / 2 + y2 * f * scale, z2];
}

function slabColor(v, max, alpha) {
  const a = Math.min(1, Math.abs(v) / (max || 1));
  const [r, g, b] = v >= 0 ? [242, 160, 75] : [91, 140, 255];
  return `rgba(${r},${g},${b},${(0.12 + 0.88 * a) * alpha})`;
}

function drawDepth() {
  const canvas = document.getElementById("depth");
  const { slabs, tokens, hover } = depthView;
  if (!slabs.length) return;
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

  // Repère : x = nombres du vecteur, y = mots, z = profondeur (ordre des couches).
  const n = slabs.length;
  const T = slabs[0].rows.length;
  const W = 1.1, H = 0.9, D = 2.2;
  const zOf = (i) => D / 2 - (n === 1 ? 0 : (i / (n - 1)) * D);
  const yOf = (t) => -H / 2 + ((t + 0.5) / T) * H;

  // Les fils d'un même mot à travers toutes les couches (le « chemin » d'un token dans BERT).
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(139,147,167,0.18)";
  for (let t = 0; t < T; t++) {
    const a = project([-W / 2 - 0.05, yOf(t), zOf(0)], cw, ch, scale);
    const b = project([-W / 2 - 0.05, yOf(t), zOf(n - 1)], cw, ch, scale);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }

  // On peint du plus loin au plus proche.
  const order = slabs.map((s, i) => ({ s, i, z: project([0, 0, zOf(i)], cw, ch, scale)[2] }))
    .sort((p, q) => q.z - p.z);
  depthView.hits = [];
  order.forEach(({ s, i }) => {
    const z = zOf(i);
    const rows = s.rows;
    const dims = rows[0].length;
    let max = 0;
    rows.forEach((r) => r.forEach((v) => { max = Math.max(max, Math.abs(v)); }));
    const alpha = hover < 0 || hover === i ? 1 : 0.35;
    const cellW = W / dims, cellH = H / T;
    rows.forEach((r, t) => {
      r.forEach((v, d) => {
        const x0 = -W / 2 + d * cellW, y0 = -H / 2 + t * cellH;
        const q = [[x0, y0], [x0 + cellW, y0], [x0 + cellW, y0 + cellH], [x0, y0 + cellH]]
          .map(([x, y]) => project([x, y, z], cw, ch, scale));
        ctx.fillStyle = slabColor(v, max, alpha);
        ctx.beginPath();
        q.forEach(([px, py], k) => (k ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
        ctx.closePath();
        ctx.fill();
      });
    });
    const corners = [[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]]
      .map(([x, y]) => project([x, y, z], cw, ch, scale));
    ctx.strokeStyle = hover === i ? "#7aa2ff" : `rgba(139,147,167,${0.5 * alpha})`;
    ctx.lineWidth = hover === i ? 2 : 1;
    ctx.beginPath();
    corners.forEach(([px, py], k) => (k ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
    ctx.closePath();
    ctx.stroke();
    depthView.hits.push({ i, corners });
  });

  // Noms des couches au-dessus des plaques, dessinés en dernier pour rester lisibles.
  ctx.font = "11px system-ui, sans-serif";
  // Les plaques serrées se chevauchent : on n'écrit un nom que s'il a la place (celui survolé, toujours).
  const placed = [];
  slabs.map((s, i) => {
    const text = `${i + 1}. ${s.name} · ${s.dim}`;
    const top = project([W / 2 + 0.04, -H / 2 - 0.04, zOf(i)], cw, ch, scale);
    const width = ctx.measureText(text).width;
    return { i, text, x: Math.min(top[0], cw - width - 4), y: top[1], width };
  }).sort((a, b) => (a.i === hover ? -1 : b.i === hover ? 1 : a.i - b.i)).forEach((l) => {
    const clash = placed.some((p) => Math.abs(p.y - l.y) < 13 && l.x < p.x + p.width + 6 && p.x < l.x + l.width + 6);
    if (clash && l.i !== hover) return;
    placed.push(l);
    ctx.fillStyle = hover === l.i ? "#e6e8ee" : `rgba(139,147,167,${hover < 0 ? 1 : 0.5})`;
    ctx.fillText(l.text, l.x, l.y);
  });

  // Les mots, devant la première plaque.
  ctx.fillStyle = "#e6e8ee";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right";
  const every = Math.ceil(T / 10);
  for (let t = 0; t < T; t++) {
    const p = project([-W / 2 - 0.06, yOf(t), zOf(0)], cw, ch, scale);
    if (tokens.length || t % every === 0) ctx.fillText(tokens[t] ?? `${t}`, p[0], p[1] + 4);
  }
  ctx.textAlign = "left";
}

function insideQuad(x, y, q) {
  let inside = false;
  for (let i = 0, j = q.length - 1; i < q.length; j = i++) {
    const [xi, yi] = q[i], [xj, yj] = q[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function slabAt(e) {
  const r = e.target.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  // hits est rangé du plus loin au plus proche : la dernière plaque touchée est devant.
  const hit = (depthView.hits || []).filter((h) => insideQuad(x, y, h.corners)).pop();
  return hit ? hit.i : -1;
}

(() => {
  const canvas = document.getElementById("depth");
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    stopSpin();
    drag = { x: e.clientX, y: e.clientY, yaw: depthView.yaw, pitch: depthView.pitch, moved: false };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      depthView.yaw = drag.yaw + dx * 0.008;
      depthView.pitch = Math.max(-1.3, Math.min(1.3, drag.pitch + dy * 0.008));
      drawDepth();
      return;
    }
    const i = slabAt(e);
    if (i !== depthView.hover) {
      depthView.hover = i;
      drawDepth();
      if (depthView.onHover) depthView.onHover(i >= 0 ? depthView.slabs[i].line : null);
    }
  });
  canvas.addEventListener("pointerup", (e) => {
    const click = drag && !drag.moved;
    drag = null;
    if (click) {
      const i = slabAt(e);
      if (i >= 0 && depthView.onPick) depthView.onPick(depthView.slabs[i].line);
    }
  });
  canvas.addEventListener("pointerleave", () => {
    if (drag || depthView.hover < 0) return;
    depthView.hover = -1;
    drawDepth();
    if (depthView.onHover) depthView.onHover(null);
  });
  window.addEventListener("resize", drawDepth);
})();
