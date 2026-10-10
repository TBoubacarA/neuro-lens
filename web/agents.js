// neuro-lens : agents LangGraph — les étapes, les embranchements, et le chemin réellement suivi.

const AGENT = { w: 132, h: 34, gapX: 22, gapY: 44, pad: 20, lane: 14 };
const stepName = (id) => (id === "__start__" ? "Départ" : id === "__end__" ? "Fin" : id);

function renderAgent(root, res, onHover, onPick, onLive) {
  root.innerHTML = "";
  const a = AGENT;
  const card = mlCard("attn", "Agent", `${res.nodes.filter((n) => n.kind === "step").length} étapes`);
  addText(card, "explain", "Chaque boîte est une étape de l'agent. Une flèche en pointillés est un choix : l'agent décide en cours de route laquelle suivre. " +
    (res.steps.length ? "En couleur : le chemin réellement suivi, avec l'ordre de passage." : ""));

  // Étage d'une étape : sa distance au départ. Une flèche qui remonte est une boucle.
  const byId = new Map(res.nodes.map((n) => [n.id, { ...n, rank: null }]));
  const queue = ["__start__"];
  if (byId.has("__start__")) byId.get("__start__").rank = 0;
  while (queue.length) {
    const id = queue.shift();
    res.edges.filter((e) => e.source === id).forEach((e) => {
      const t = byId.get(e.target);
      if (t && t.rank === null) { t.rank = byId.get(id).rank + 1; queue.push(e.target); }
    });
  }
  const nodes = [...byId.values()];
  nodes.forEach((n) => { n.rank ??= 0; });
  const end = byId.get("__end__");
  if (end) end.rank = Math.max(...nodes.map((n) => n.rank), 0); // la fin tout en bas
  const ranks = [];
  nodes.forEach((n) => (ranks[n.rank] ??= []).push(n));
  const rows = ranks.filter(Boolean);
  const widest = Math.max(...rows.map((r) => r.length));
  const inner = widest * a.w + (widest - 1) * a.gapX;
  const loops = res.edges.filter((e) => byId.get(e.target).rank <= byId.get(e.source).rank);
  const left = a.pad + (loops.length ? 10 + loops.length * a.lane : 0);
  rows.forEach((row, r) => row.forEach((n, k) => {
    const rowW = row.length * a.w + (row.length - 1) * a.gapX;
    n.x = left + (inner - rowW) / 2 + k * (a.w + a.gapX);
    n.y = a.pad + r * (a.h + a.gapY);
  }));
  const W = left + inner + a.pad, H = a.pad * 2 + rows.length * a.h + (rows.length - 1) * a.gapY;

  // Chemin suivi : départ -> étapes dans l'ordre -> fin (si l'agent est allé au bout).
  const path = res.steps.map((s) => s.node);
  const walked = new Set();
  if (path.length) {
    walked.add(`__start__>${path[0]}`);
    path.forEach((p, i) => { if (i) walked.add(`${path[i - 1]}>${p}`); });
    if (!res.stopped) walked.add(`${path[path.length - 1]}>__end__`);
  }

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "agent-svg" });
  svg.appendChild(el("defs", {})).innerHTML = ["", "-on"].map((s) =>
    `<marker id="nl-aarrow${s}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">` +
    `<path d="M0,0 L8,4 L0,8 z" class="agent-arrow${s}"/></marker>`).join("");
  let lane = 0;
  res.edges.forEach((e) => {
    const s = byId.get(e.source), t = byId.get(e.target);
    const on = walked.has(`${e.source}>${e.target}`);
    let d, lx, ly;
    if (t.rank > s.rank) {
      const sx = s.x + a.w / 2, sy = s.y + a.h, tx = t.x + a.w / 2, ty = t.y - 1;
      d = `M${sx},${sy} C${sx},${sy + 20} ${tx},${ty - 20} ${tx},${ty}`;
      [lx, ly] = [(sx + tx) / 2, (sy + ty) / 2];
    } else { // boucle : on remonte par la gauche
      // On sort par le bas de la boîte pour passer sous ses voisines, pas au travers.
      const x = left - 12 - lane * a.lane, sx = s.x + a.w / 2, sy = s.y + a.h + 10 + lane++ * 5, ty = t.y + a.h / 2;
      d = `M${sx},${s.y + a.h} V${sy} H${x + 8} Q${x},${sy} ${x},${sy - 8} V${ty + 8} Q${x},${ty} ${x + 8},${ty} H${t.x - 1}`;
      [lx, ly] = [x, (sy + ty) / 2];
    }
    svg.appendChild(el("path", {
      d, class: `agent-edge${e.conditional ? " choice" : ""}${on ? " on" : ""}`,
      "marker-end": `url(#nl-aarrow${on ? "-on" : ""})`,
    }));
    if (e.label) {
      const text = el("text", { x: lx, y: ly + 3, class: "agent-edge-label" }, e.label);
      svg.appendChild(text);
    }
  });
  nodes.forEach((n) => {
    const visits = path.map((p, i) => (p === n.id ? i + 1 : 0)).filter(Boolean);
    const g = svg.appendChild(el("g", { class: `agent-node ${n.kind}${visits.length ? " on" : ""}`, "data-node": n.id }));
    if (n.line) g.dataset.line = n.line;
    g.appendChild(el("rect", { x: n.x, y: n.y, width: a.w, height: a.h, rx: n.kind === "step" ? 6 : 17 }));
    g.appendChild(el("text", { x: n.x + a.w / 2, y: n.y + 21, class: "blocks-label" }, stepName(n.id)));
    if (visits.length) {
      g.appendChild(el("text", { x: n.x + a.w - 4, y: n.y - 4, class: "agent-order" },
        `passage ${visits.slice(0, 6).join(", ")}${visits.length > 6 ? "…" : ""}`));
    }
    g.addEventListener("mouseenter", () => onHover(n.line));
    g.addEventListener("mouseleave", () => onHover(null));
    g.addEventListener("click", () => onPick(n.line));
  });
  const box = document.createElement("div");
  box.className = "graph";
  box.appendChild(svg);
  card.appendChild(box);
  if (res.hint) addText(card, "meta", res.hint);
  if (res.llm) {
    const bar = document.createElement("div");
    bar.className = "llm-bar";
    const go = document.createElement("button");
    go.className = "small";
    go.textContent = "▶ Lancer avec Claude";
    go.disabled = !res.llm.key;
    go.addEventListener("click", () => { go.disabled = true; onLive(); });
    bar.appendChild(go);
    const note = document.createElement("span");
    note.className = "meta";
    note.textContent = !res.llm.key
      ? "Pour que Claude pilote l'agent, lance le serveur avec ta clé : ANTHROPIC_API_KEY=… python -m server.app. Sans clé, une règle fixe joue son rôle."
      : res.llm.live
        ? "Déroulé réel : chaque passage par « reflechir » est un appel à Claude, qui a choisi lui-même ses outils."
        : "Déroulé ci-dessous : la règle fixe. Clique pour que Claude décide lui-même (un appel payant par passage).";
    bar.appendChild(note);
    card.appendChild(bar);
  }
  root.appendChild(card);
  if (!res.start) return;

  const run = mlCard("dense", "Déroulé", `${res.steps.length} passage${res.steps.length > 1 ? "s" : ""}`);
  addText(run, "explain", "Ce que chaque étape a changé dans l'état de l'agent (sa mémoire de travail), dans l'ordre. Survole une ligne pour retrouver l'étape dans le schéma et dans le code.");
  const list = document.createElement("div");
  list.className = "steps";
  const row = (num, name, values, node) => {
    const r = document.createElement("div");
    r.innerHTML = `<b></b><span class="step-name"></span><dl></dl>`;
    r.querySelector("b").textContent = num;
    r.querySelector(".step-name").textContent = name;
    Object.entries(values).forEach(([k, v]) => {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = k; dd.textContent = v;
      r.querySelector("dl").append(dt, dd);
    });
    if (!Object.keys(values).length) r.querySelector("dl").textContent = "rien de changé";
    if (node) {
      r.addEventListener("mouseenter", () => { markNode(svg, node.id); onHover(node.line); });
      r.addEventListener("mouseleave", () => { markNode(svg, null); onHover(null); });
      r.addEventListener("click", () => onPick(node.line));
    }
    list.appendChild(r);
  };
  row("", "État de départ", res.start);
  res.steps.forEach((s, i) => row(i + 1, s.node, s.changes, byId.get(s.node)));
  row("", "État final", res.final);
  run.appendChild(list);
  if (res.stopped) addText(run, "meta", res.stopped).classList.add("stopped");
  root.appendChild(run);
}

function markNode(svg, id) {
  svg.querySelectorAll(".agent-node").forEach((n) => n.classList.toggle("active", n.dataset.node === id));
}
