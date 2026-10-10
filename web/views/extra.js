// Vues « ML classique » (scikit-learn) et « agents IA » (LangGraph).
// Elles prennent la place des cartes PyTorch quand le serveur renvoie kind = "sklearn" ou "agent".

(function () {
  const COLORS = ["#5b8cff", "#f2b84b", "#2fbfa5", "#ff6b6b", "#b083f0", "#7dd3fc", "#f472b6", "#a3e635"];
  const SVG = "http://www.w3.org/2000/svg";
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  let root = null;
  let replayTimer = null;

  function container() {
    if (!root) {
      root = document.createElement("div");
      root.id = "alt-view";
      document.getElementById("lens").appendChild(root);
    }
    return root;
  }

  function show(on) {
    document.getElementById("lens").classList.toggle("alt", on);
    if (!on && root) root.innerHTML = "";
  }

  function el(tag, attrs = {}, html = "") {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (html) e.innerHTML = html;
    return e;
  }

  function svg(tag, attrs = {}) {
    const e = document.createElementNS(SVG, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  }

  function card(title, family, line, body) {
    const c = el("div", { class: "card", "data-family": family, "data-line": line ?? "" });
    c.innerHTML = `<div class="card-head"><span class="tag">${title}</span>
      ${line ? `<span class="line-ref">ligne ${line}</span>` : ""}</div>`;
    if (body) c.appendChild(body);
    if (line) {
      c.addEventListener("mouseenter", () => window.highlightLine && highlightLine(line));
      c.addEventListener("mouseleave", () => window.highlightLine && highlightLine(null));
    }
    return c;
  }

  function note(text) { return el("p", { class: "explain" }, esc(text)); }

  // ---------------- ML classique ----------------

  function renderSklearn(res) {
    const r = container();
    r.innerHTML = "";
    const head = el("div");
    head.appendChild(note(res.explain || ""));
    const chips = Object.entries(res.params)
      .filter(([, v]) => v !== null && v !== "" && v !== "deprecated")
      .slice(0, 10)
      .map(([k, v]) => `<span class="chip">${esc(k)} = ${esc(v)}</span>`).join("");
    head.appendChild(el("div", { class: "chips" }, chips));
    const facts = [
      `Tâche : ${res.task}`,
      `${res.n_samples} exemples · ${res.features.length} caractéristiques`,
      res.score ? `${res.score.name} : ${res.score.value}` : null,
      res.fitted_here ? "entraîné ici sur X, y" : null,
    ].filter(Boolean).join(" · ");
    head.appendChild(el("div", { class: "meta" }, esc(facts)));
    r.appendChild(card(res.estimator, "dense", res.line, head));

    res.views.forEach((v) => {
      if (v.kind === "pipeline") r.appendChild(card("Pipeline", "other", null, pipelineView(v)));
      if (v.kind === "tree") r.appendChild(card(`Arbre de décision · profondeur ${v.depth} · ${v.leaves} feuilles`, "pool", res.line, treeView(v, res)));
      if (v.kind === "plane") r.appendChild(card(v.projected ? "Données projetées en 2D" : v.centers ? "Groupes trouvés" : "Frontière de décision", "conv", res.fit_line, planeView(v, res)));
      if (v.kind === "bars") r.appendChild(card(v.title, "act", null, barsView(v)));
    });
  }

  function pipelineView(v) {
    return el("div", { class: "flow" }, v.steps.map(([n, t]) => `<span class="chip">${esc(n)}<small>${esc(t)}</small></span>`).join('<span class="arrow">→</span>'));
  }

  function treeView(v, res) {
    const wrap = el("div", { class: "tree-wrap" });
    if (v.note) wrap.appendChild(note(v.note));
    // Placement : chaque feuille dessinée prend une colonne, un parent se centre sur ses enfants.
    const W = 128, H = 54, GX = 12, GY = 34;
    let col = 0;
    const pos = [];
    (function place(i) {
      const n = v.nodes[i];
      if (n.children) {
        n.children.forEach(place);
        pos[i] = (pos[n.children[0]] + pos[n.children[1]]) / 2;
      } else pos[i] = col++;
    })(0);
    const depth = Math.max(...v.nodes.map((n) => n.depth)) + 1;
    const width = col * (W + GX), height = depth * (H + GY);
    const s = svg("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", style: `max-width:${width}px;min-width:${Math.min(width, 520)}px`, class: "tree" });
    const xy = (i) => [pos[i] * (W + GX), v.nodes[i].depth * (H + GY)];
    v.nodes.forEach((n, i) => (n.children || []).forEach((c, k) => {
      const [x1, y1] = xy(i), [x2, y2] = xy(c);
      s.appendChild(svg("path", { d: `M${x1 + W / 2},${y1 + H} C${x1 + W / 2},${y1 + H + GY / 2} ${x2 + W / 2},${y2 - GY / 2} ${x2 + W / 2},${y2}`, class: "edge" }));
      const t = svg("text", { x: (x1 + x2 + W) / 2 + (k ? 6 : -6), y: y1 + H + GY / 2 + 4, class: "edge-label", "text-anchor": k ? "start" : "end" });
      t.textContent = k ? "non" : "oui";
      s.appendChild(t);
    }));
    v.nodes.forEach((n, i) => {
      const [x, y] = xy(i);
      const color = n.class !== undefined ? COLORS[n.class % COLORS.length] : "#7aa2ff";
      const g = svg("g", { transform: `translate(${x},${y})` });
      g.appendChild(svg("rect", { width: W, height: H, rx: 6, class: n.leaf ? "node leaf" : "node", style: `stroke:${color}` }));
      const t1 = svg("text", { x: W / 2, y: 18, "text-anchor": "middle", class: "node-title" });
      t1.textContent = n.leaf ? `→ ${n.label}` : n.question;
      const t2 = svg("text", { x: W / 2, y: 34, "text-anchor": "middle", class: "node-sub" });
      t2.textContent = `${n.samples} ex.${n.truncated ? " · …" : ""}`;
      g.append(t1, t2);
      if (n.dist) {
        let off = 8;
        n.dist.forEach((d, k) => {
          const w = d * (W - 16);
          if (w > 0) g.appendChild(svg("rect", { x: off, y: 41, width: w, height: 6, fill: COLORS[k % COLORS.length] }));
          off += w;
        });
      }
      const title = svg("title");
      title.textContent = `impureté ${n.impurity}${n.dist ? " · répartition " + n.dist.join(" / ") : ""}`;
      g.appendChild(title);
      s.appendChild(g);
    });
    wrap.appendChild(s);
    if (res.classes.length) wrap.appendChild(legend(res.classes));
    return wrap;
  }

  function legend(classes) {
    return el("div", { class: "legend" }, classes.map((c, k) => `<span><i style="background:${COLORS[k % COLORS.length]}"></i>${esc(c)}</span>`).join(""));
  }

  function planeView(v, res) {
    const wrap = el("div");
    const size = 340, pad = 28;
    const cv = el("canvas", { width: size, height: size, class: "plane" });
    const ctx = cv.getContext("2d");
    const xs = v.points.map((p) => p.x), ys = v.points.map((p) => p.y);
    const xr = v.grid ? v.grid.x : [Math.min(...xs), Math.max(...xs)];
    const yr = v.grid ? v.grid.y : [Math.min(...ys), Math.max(...ys)];
    const sx = (x) => pad + ((x - xr[0]) / (xr[1] - xr[0] || 1)) * (size - 2 * pad);
    const sy = (y) => size - pad - ((y - yr[0]) / (yr[1] - yr[0] || 1)) * (size - 2 * pad);
    const labels = [...new Set(v.points.map((p) => p.label))];
    const classIndex = (l) => {
      const i = res.classes.indexOf(String(l));
      return i >= 0 ? i : typeof l === "number" && Number.isInteger(l) ? ((l % COLORS.length) + COLORS.length) % COLORS.length : labels.indexOf(l);
    };
    ctx.fillStyle = "#0f1117";
    ctx.fillRect(0, 0, size, size);
    if (v.grid) {
      const n = v.grid.values.length, cw = (size - 2 * pad) / n;
      const flat = v.grid.values.flat();
      const lo = Math.min(...flat), hi = Math.max(...flat);
      v.grid.values.forEach((row, j) => row.forEach((val, i) => {
        if (v.grid.discrete) {
          ctx.fillStyle = COLORS[classIndex(val) % COLORS.length] + "40";
        } else {
          const t = (val - lo) / (hi - lo || 1);
          ctx.fillStyle = `rgba(${Math.round(40 + 215 * t)},${Math.round(70 + 150 * t)},${Math.round(160 - 70 * t)},0.45)`;
        }
        ctx.fillRect(pad + i * cw, size - pad - (j + 1) * cw, cw + 0.5, cw + 0.5);
      }));
    }
    const continuous = !res.classes.length && res.task === "régression";
    const lv = v.points.map((p) => p.label).filter((l) => typeof l === "number");
    const lmin = Math.min(...lv), lmax = Math.max(...lv);
    v.points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(sx(p.x), sy(p.y), 3.5, 0, Math.PI * 2);
      if (continuous && typeof p.label === "number") {
        const t = (p.label - lmin) / (lmax - lmin || 1);
        ctx.fillStyle = `rgb(${Math.round(40 + 215 * t)},${Math.round(70 + 150 * t)},${Math.round(160 - 70 * t)})`;
      } else ctx.fillStyle = p.label === -1 ? "#7d8597" : COLORS[classIndex(p.label) % COLORS.length];
      ctx.fill();
      ctx.strokeStyle = "#0f1117";
      ctx.stroke();
    });
    (v.centers || []).forEach(([x, y]) => {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(sx(x) - 7, sy(y) - 7); ctx.lineTo(sx(x) + 7, sy(y) + 7);
      ctx.moveTo(sx(x) + 7, sy(y) - 7); ctx.lineTo(sx(x) - 7, sy(y) + 7);
      ctx.stroke();
      ctx.lineWidth = 1;
    });
    ctx.fillStyle = "#8b93a7";
    ctx.font = "11px system-ui";
    ctx.fillText(v.axes[0], size / 2 - 20, size - 8);
    ctx.save(); ctx.translate(12, size / 2 + 20); ctx.rotate(-Math.PI / 2); ctx.fillText(v.axes[1], 0, 0); ctx.restore();
    wrap.appendChild(note(v.projected
      ? "Tes données ont plus de 2 caractéristiques : on les aplatit sur les 2 axes qui gardent le plus d'information pour les voir."
      : v.centers ? "Chaque zone colorée appartient à un centre (×) : le modèle a trouvé ces groupes tout seul."
      : v.grid && v.grid.discrete ? "Chaque zone colorée est la classe que le modèle prédirait à cet endroit ; les points sont les vraies données."
      : "La couleur du fond est la valeur que le modèle prédit à cet endroit."));
    wrap.appendChild(cv);
    if (res.classes.length) wrap.appendChild(legend(res.classes));
    return wrap;
  }

  function barsView(v) {
    const wrap = el("div", { class: "hbars" });
    wrap.appendChild(note(v.note));
    const max = Math.max(...v.values.map(Math.abs)) || 1;
    v.labels.forEach((lab, i) => {
      const val = v.values[i];
      const row = el("div", { class: "hbar" }, `<span class="hbar-label">${esc(lab)}</span>
        <span class="hbar-track"><i style="width:${(Math.abs(val) / max) * 100}%;background:${val >= 0 ? "var(--dense)" : "var(--danger)"}"></i></span>
        <span class="hbar-val">${val}</span>`);
      wrap.appendChild(row);
    });
    return wrap;
  }

  // ---------------- Agents IA ----------------

  function renderAgent(res) {
    const r = container();
    r.innerHTML = "";
    clearInterval(replayTimer);
    const graphBody = el("div");
    graphBody.appendChild(note("Chaque boîte est une étape de l'agent ; les flèches en pointillés sont des choix que l'agent fait pendant qu'il tourne."));
    const g = agentGraph(res);
    graphBody.appendChild(g.svg);
    graphBody.appendChild(el("div", { class: "legend" },
      `<span><i style="background:var(--ok)"></i>agent (décide)</span><span><i style="background:var(--dense)"></i>outil</span><span><i style="background:var(--other)"></i>début / fin</span>`));
    r.appendChild(card("Graphe de l'agent", "dense", null, graphBody));

    const traceBody = el("div");
    if (res.note) traceBody.appendChild(note(res.note));
    if (res.inputs) traceBody.appendChild(el("div", { class: "meta" }, `Entrée : <code>${esc(res.inputs)}</code>`));
    const list = el("ol", { class: "trace" });
    res.trace.forEach((step, i) => {
      const upd = typeof step.update === "string" ? esc(step.update)
        : Object.entries(step.update).map(([k, v]) => `<code>${esc(k)}</code> = ${esc(v)}`).join("<br>");
      const li = el("li", { "data-line": step.line ?? "" }, `<b>${esc(step.node)}</b><div>${upd || "<i>rien de modifié</i>"}</div>`);
      li.addEventListener("mouseenter", () => { g.focus(i); window.highlightLine && highlightLine(step.line); });
      list.appendChild(li);
    });
    traceBody.appendChild(list);
    if (res.trace.length) {
      const btn = el("button", {}, "▶ Rejouer le chemin");
      btn.addEventListener("click", () => replay(res, g, list));
      traceBody.prepend(btn);
    }
    if (!res.ok) traceBody.appendChild(el("div", { class: "explain err-text" }, `✗ ${esc(res.error)}`));
    r.appendChild(card(`Chemin suivi · ${res.trace.length} étapes`, "act", null, traceBody));

    if (res.final_state) {
      const rows = Object.entries(res.final_state).map(([k, v]) => `<tr><td><code>${esc(k)}</code></td><td>${esc(v)}</td></tr>`).join("");
      r.appendChild(card("État final", "other", null, el("table", { class: "state" }, rows)));
    }
  }

  function replay(res, g, list) {
    clearInterval(replayTimer);
    let i = 0;
    const items = [...list.children];
    const tick = () => {
      items.forEach((li, k) => li.classList.toggle("active", k === i));
      g.focus(i);
      window.highlightLine && highlightLine(res.trace[i]?.line);
      i += 1;
      if (i > res.trace.length) {
        clearInterval(replayTimer);
        items.forEach((li) => li.classList.remove("active"));
        g.focus(-1);
        window.highlightLine && highlightLine(null);
      }
    };
    tick();
    replayTimer = setInterval(tick, 900);
  }

  function agentGraph(res) {
    // Niveaux : parcours en largeur depuis le début ; les flèches qui remontent sont des boucles.
    const level = { __start__: 0 };
    const queue = ["__start__"];
    while (queue.length) {
      const n = queue.shift();
      res.edges.filter((e) => e.source === n).forEach((e) => {
        if (level[e.target] === undefined) { level[e.target] = level[n] + 1; queue.push(e.target); }
      });
    }
    res.nodes.forEach((n) => { if (level[n.id] === undefined) level[n.id] = 1; });
    if (level.__end__ !== undefined) { // la fin tout en bas, seule sur sa ligne
      delete level.__end__;
      level.__end__ = Math.max(...Object.values(level)) + 1;
    }
    const rows = {};
    res.nodes.forEach((n) => (rows[level[n.id]] = rows[level[n.id]] || []).push(n));
    const W = 130, H = 38, GY = 62, width = 460;
    const pos = {};
    Object.entries(rows).forEach(([lv, ns]) => ns.forEach((n, k) => {
      pos[n.id] = { x: (width / (ns.length + 1)) * (k + 1) - W / 2, y: 16 + lv * (H + GY) };
    }));
    const height = Math.max(...Object.values(pos).map((p) => p.y)) + H + 20;
    const s = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "agent-graph", width: "100%", style: `max-width:${width}px` });
    const defs = svg("defs");
    const marker = svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
    marker.appendChild(svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "context-stroke" }));
    defs.appendChild(marker);
    s.appendChild(defs);
    const edgeEls = [];
    res.edges.forEach((e) => {
      const a = pos[e.source], b = pos[e.target];
      if (!a || !b) return;
      const up = b.y <= a.y;
      const skip = !up && b.y - a.y > H + GY + 1; // saute un niveau : on passe par la gauche
      let d;
      if (skip) {
        const x1 = a.x, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2;
        d = `M${x1},${y1} C${x1 - 70},${y1} ${x2 - 70},${y2} ${x2 - 2},${y2}`;
      } else if (up) { // boucle : on contourne par la droite
        const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x + W, y2 = b.y + H / 2;
        d = `M${x1},${y1} C${x1 + 60},${y1} ${x2 + 60},${y2} ${x2 + 2},${y2}`;
      } else {
        d = `M${a.x + W / 2},${a.y + H} L${b.x + W / 2},${b.y - 2}`;
      }
      const p = svg("path", { d, class: `a-edge${e.conditional ? " cond" : ""}`, "marker-end": "url(#arrow)" });
      s.appendChild(p);
      edgeEls.push({ e, p });
      if (e.label) {
        const t = svg("text", { class: "edge-label", x: up ? Math.max(a.x, b.x) + W + 40 : skip ? Math.min(a.x, b.x) - 66 : (a.x + b.x + W) / 2 + 6, y: up ? (a.y + b.y + H) / 2 : (a.y + H + b.y) / 2 + 4 });
        t.textContent = e.label;
        s.appendChild(t);
      }
    });
    const nodeEls = {};
    res.nodes.forEach((n) => {
      const { x, y } = pos[n.id];
      const g = svg("g", { transform: `translate(${x},${y})`, class: `a-node ${n.kind}` });
      g.appendChild(svg("rect", { width: W, height: H, rx: n.kind === "start" || n.kind === "end" ? H / 2 : 8 }));
      const t = svg("text", { x: W / 2, y: H / 2 + 5, "text-anchor": "middle" });
      t.textContent = n.label;
      g.appendChild(t);
      const line = n.def_line || n.line;
      if (line) {
        g.style.cursor = "pointer";
        g.addEventListener("mouseenter", () => window.highlightLine && highlightLine(line));
        g.addEventListener("mouseleave", () => window.highlightLine && highlightLine(null));
      }
      s.appendChild(g);
      nodeEls[n.id] = g;
    });
    return {
      svg: s,
      focus(i) {
        Object.values(nodeEls).forEach((g) => g.classList.remove("on"));
        edgeEls.forEach(({ p }) => p.classList.remove("on"));
        const step = res.trace[i];
        if (!step) return;
        nodeEls[step.node]?.classList.add("on");
        const prev = i === 0 ? "__start__" : res.trace[i - 1].node;
        edgeEls.filter(({ e }) => e.source === prev && e.target === step.node).forEach(({ p }) => p.classList.add("on"));
      },
    };
  }

  // ---------------- Branchement ----------------

  window.NeuroViews = {
    // Renvoie true si la réponse a été dessinée ici.
    handle(res) {
      if (res.kind === "sklearn" || res.kind === "agent") {
        show(true);
        res.kind === "sklearn" ? renderSklearn(res) : renderAgent(res);
        return true;
      }
      if (res.ok) show(false); // modèle PyTorch : les cartes habituelles reprennent la main
      return false;
    },
  };
})();
