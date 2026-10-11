// neuro-lens : LLMOps et AgentOps — la trace d'un agent appel par appel, ses tokens et son coût,
// le garde-fou, et son évaluation sur un jeu de cas. (Le rôle de Langfuse ou LangSmith en production.)

const fmtMs = (ms) => (ms < 10 ? `${ms.toFixed(1)} ms` : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const fmtCost = (c) => (c == null ? "—" : c === 0 ? "0 $" : c < 0.01 ? `${(c * 100).toFixed(2)} ¢` : `${c.toFixed(3)} $`);
const fmtTok = (n) => n.toLocaleString("fr-FR");

function renderLLMOps(root, res, onHover, onPick) {
  const ops = res.ops;
  if (!ops) return;
  const lineOf = new Map(res.nodes.map((n) => [n.id, n.line]));

  // 1. La trace : chaque étape et chaque appel au modèle, sur une ligne de temps.
  const card = mlCard("conv", "Trace", ops.calls ? `${ops.calls} appel${ops.calls > 1 ? "s" : ""} au modèle` : "aucun appel au modèle");
  addText(card, "explain", ops.calls
    ? "Chaque barre est une étape de l'agent ; en couleur, les appels au modèle de langage qu'elle contient, avec leurs tokens. C'est ce qu'un outil de trace enregistre en production pour retrouver pourquoi un agent a répondu ça."
    : "Chaque barre est une étape de l'agent et sa durée. Ici le modèle est joué par une règle fixe : aucun appel, aucun token. Clique sur « Lancer avec Claude » pour voir les vrais appels, leurs tokens et leur coût.");
  const kpis = document.createElement("div");
  kpis.className = "llm-kpis";
  const kpi = (label, value, note) => {
    const d = document.createElement("div");
    d.innerHTML = `<span></span><b></b><i></i>`;
    d.querySelector("span").textContent = label;
    d.querySelector("b").textContent = value;
    d.querySelector("i").textContent = note || "";
    kpis.appendChild(d);
  };
  kpi("durée", fmtMs(ops.duration), `${ops.steps} étape${ops.steps > 1 ? "s" : ""}`);
  kpi("appels au modèle", String(ops.calls), ops.limits ? `limite : ${ops.limits.appels}` : "");
  kpi("tokens", `${fmtTok(ops.tokens_in)} → ${fmtTok(ops.tokens_out)}`, ops.cached ? `${fmtTok(ops.cached)} lus en cache` : "entrée → sortie");
  kpi("coût", fmtCost(ops.cost), ops.cost == null ? "ajoute PRIX au code" : ops.priced ? "d'après PRIX" : "");
  card.appendChild(kpis);
  if (ops.stopped) addText(card, "meta", ops.stopped).classList.add("llm-stop");
  card.appendChild(waterfall(ops.spans, ops.duration, lineOf, onHover, onPick));
  const toolNames = Object.entries(ops.tools);
  if (toolNames.length) {
    addText(card, "meta", "Outils demandés par le modèle : " + toolNames.map(([k, v]) => `${k} ×${v}`).join(", "));
  }
  root.appendChild(card);

  // 2. L'évaluation : l'agent est noté sur un jeu de cas.
  const ev = ops.eval;
  if (ev === null || ev === undefined) {
    const hint = mlCard("dense", "Évaluation", "pas de cas");
    addText(hint, "explain", "Définis `cas`, une liste de {question, attendu} : neuro-lens fera tourner l'agent sur chacun et le notera. C'est ainsi qu'on compare deux prompts ou deux modèles avant de choisir.");
    root.appendChild(hint);
    return;
  }
  const grade = mlCard("dense", "Évaluation", `${ev.passed} sur ${ev.total} réussi${ev.passed > 1 ? "s" : ""}`);
  grade.querySelector(".shape").classList.add(ev.passed === ev.total ? "llm-good" : "llm-bad");
  addText(grade, "explain", "Pour chaque cas, l'agent repart de zéro avec la question du cas ; il réussit si la réponse attendue se trouve dans sa réponse. Change le prompt (`system`), le modèle ou les outils, et regarde la note bouger : c'est la boucle d'une équipe LLMOps.");
  const totalCost = ev.rows.reduce((s, r) => s + (r.cost ?? 0), 0);
  const bar = document.createElement("div");
  bar.className = "llm-score";
  bar.innerHTML = `<div class="llm-score-fill"></div>`;
  bar.querySelector("div").style.width = `${ev.total ? (100 * ev.passed) / ev.total : 0}%`;
  bar.querySelector("div").classList.add(ev.passed === ev.total ? "good" : ev.passed ? "mid" : "bad");
  grade.appendChild(bar);
  const table = document.createElement("table");
  table.className = "llm-table";
  table.innerHTML = `<thead><tr><th></th><th>cas</th><th>attendu</th><th>obtenu</th><th>passages</th><th>appels</th><th>tokens</th><th>coût</th><th>durée</th></tr></thead><tbody></tbody>`;
  const body = table.querySelector("tbody");
  ev.rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = r.ok ? "ok" : "ko";
    [r.ok ? "✓" : "✗", r.input, r.expected, r.stopped ? `${r.got || ""} (${r.stopped})` : r.got, r.steps, r.calls, fmtTok(r.tokens), fmtCost(r.cost), fmtMs(r.duration)]
      .forEach((v, i) => { const td = document.createElement("td"); td.textContent = v; if (i >= 4) td.className = "num"; tr.appendChild(td); });
    body.appendChild(tr);
  });
  grade.appendChild(table);
  const note = [`Coût total de l'évaluation : ${fmtCost(ev.rows.some((r) => r.cost == null) ? null : totalCost)}.`];
  if (ev.skipped) note.push(`${ev.skipped} cas de plus non joués (8 au maximum).`);
  if (!ops.calls) note.push("La règle fixe répond toujours la même chose : c'est pour ça qu'elle rate les autres questions.");
  addText(grade, "meta", note.join(" "));
  root.appendChild(grade);
}

// Une ligne par étape, les appels au modèle en dessous, le temps de gauche à droite.
function waterfall(spans, total, lineOf, onHover, onPick) {
  const W = 560, left = 118, rowH = 18, pad = 6;
  const rows = [];
  spans.forEach((s, i) => rows.push({ s, i }));
  const H = pad * 2 + rows.length * rowH + 14;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "llm-svg", width: W, height: H });
  const x = (t) => left + (total ? (t / total) * (W - left - 28) : 0);
  // Repères de temps.
  const ticks = 4;
  for (let k = 0; k <= ticks; k++) {
    const t = (total * k) / ticks, px = x(t);
    svg.appendChild(el("line", { x1: px, y1: pad, x2: px, y2: H - 14, class: "llm-grid" }));
    svg.appendChild(el("text", { x: px, y: H - 3, class: "llm-tick" }, fmtMs(t)));
  }
  rows.forEach(({ s }, r) => {
    const y = pad + r * rowH;
    const g = svg.appendChild(el("g", { class: `llm-span ${s.kind}` }));
    const label = s.kind === "llm" ? `  ↳ ${s.model.replace(/^claude-/, "")}` : s.name;
    g.appendChild(el("text", { x: left - 6, y: y + 12, class: "llm-name" }, label.length > 20 ? label.slice(0, 19) + "…" : label));
    const x0 = x(s.start), x1 = Math.max(x(s.end), x0 + 2);
    g.appendChild(el("rect", { x: x0, y: y + 3, width: x1 - x0, height: rowH - 7, rx: 3 }));
    let right = "";
    if (s.kind === "llm") right = `${fmtTok(s.tokens_in)} → ${fmtTok(s.tokens_out)} tok${s.tools.length ? " · " + s.tools.join(", ") : ""}${s.stop && s.stop !== "end_turn" && s.stop !== "tool_use" ? " · " + s.stop : ""}`;
    else if (s.changed && s.changed.length) right = `${fmtMs(s.end - s.start)} · ${s.changed.join(", ")}`;
    else right = fmtMs(s.end - s.start);
    const tx = x1 + 5, fits = tx + right.length * 5.3 < W - 4;
    g.appendChild(el("text", { x: fits ? tx : x0 - 5, y: y + 12, class: `llm-detail${fits ? "" : " end"}` }, right));
    const tip = s.kind === "llm"
      ? `${s.model} · ${s.messages} message${s.messages > 1 ? "s" : ""} envoyés · ${s.tokens_in} tokens en entrée, ${s.tokens_out} en sortie${s.cached ? ` (${s.cached} lus en cache)` : ""} · arrêt : ${s.stop}${s.text ? `\n« ${s.text} »` : ""}`
      : `étape ${s.name} · ${fmtMs(s.end - s.start)}${s.changed && s.changed.length ? ` · a changé : ${s.changed.join(", ")}` : ""}`;
    g.appendChild(el("title", {}, tip));
    const line = s.kind === "step" ? lineOf.get(s.name) : null;
    if (line) {
      g.classList.add("link");
      g.addEventListener("mouseenter", () => onHover(line));
      g.addEventListener("mouseleave", () => onHover(null));
      g.addEventListener("click", () => onPick(line));
    }
  });
  const box = document.createElement("div");
  box.className = "graph llm-box";
  box.appendChild(svg);
  return box;
}
