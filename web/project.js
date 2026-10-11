// neuro-lens : la vue « Projet » — l'explorateur de fichiers à gauche du code, comme dans VS Code.
// Un clic ouvre le fichier dans l'éditeur (en lecture) ; à droite, son rôle, son plan, ses dépendances.

window.projectMode = false;
const proj = { root: "app", data: null, open: null, example: null, exampleName: "Exemple", openDirs: new Set(["server", "web", "tests"]) };

const FILE_ICON = { py: "🐍", js: "🟨", html: "🌐", css: "🎨", md: "📝", json: "{}", txt: "≡", yml: "⚙", yaml: "⚙", joblib: "📦", Dockerfile: "🐳", gitignore: "∅" };
const iconOf = (name) => FILE_ICON[name] || FILE_ICON[name.split(".").pop()] || "▫";
const LANG = { py: "python", js: "javascript", html: "html", css: "css", md: "markdown", json: "json", yml: "yaml", yaml: "yaml", txt: "plaintext", Dockerfile: "dockerfile" };

async function toggleExplorer() {
  const side = $("explorer");
  side.hidden = !side.hidden;
  $("project-btn").classList.toggle("on", !side.hidden);
  if (!side.hidden && !proj.data) await loadProject(proj.root);
}

async function loadProject(root) {
  proj.root = root;
  const r = await fetch(`/api/project?root=${encodeURIComponent(root)}`);
  proj.data = await r.json();
  if (!proj.data.ok) { $("project-tree").textContent = proj.data.error; return; }
  const sel = $("project-root");
  sel.innerHTML = "";
  proj.data.roots.forEach((x) => sel.add(new Option(x.name, x.id, false, x.id === root)));
  drawTree();
}

function drawTree() {
  const box = $("project-tree");
  box.innerHTML = "";
  const walk = (nodes, depth) => {
    nodes.forEach((n) => {
      const row = document.createElement("div");
      row.className = `tree-row ${n.kind}${proj.open && proj.open.path === n.path ? " on" : ""}`;
      row.style.paddingLeft = `${8 + depth * 14}px`;
      row.title = n.role + (n.why ? `\n${n.why}` : "");
      const isOpen = n.kind === "dir" && proj.openDirs.has(n.path);
      row.innerHTML = `<span class="tree-icon"></span><span class="tree-name"></span><span class="tree-meta"></span>`;
      row.querySelector(".tree-icon").textContent = n.kind === "dir" ? (isOpen ? "▾" : "▸") : iconOf(n.name);
      row.querySelector(".tree-name").textContent = n.name;
      row.querySelector(".tree-meta").textContent = n.kind === "file" && n.lines ? `${n.lines}` : "";
      row.addEventListener("click", () => {
        if (n.kind === "dir") { proj.openDirs.has(n.path) ? proj.openDirs.delete(n.path) : proj.openDirs.add(n.path); drawTree(); }
        else openFile(n);
      });
      box.appendChild(row);
      if (isOpen) walk(n.children, depth + 1);
    });
  };
  walk(proj.data.tree, 0);
}

async function openFile(node) {
  const r = await fetch(`/api/project/file?root=${encodeURIComponent(proj.root)}&path=${encodeURIComponent(node.path)}`);
  const info = await r.json();
  if (!info.ok) { setStatus(info.error, "err"); return; }
  if (!window.projectMode) { proj.example = getCode(); proj.exampleName = $("examples").value; }
  window.projectMode = true;
  proj.open = { ...node, ...info };
  const ext = node.name.includes(".") ? node.name.split(".").pop() : node.name;
  if (monacoRef) {
    monacoRef.editor.setModelLanguage(editor.getModel(), LANG[ext] || "plaintext");
    editor.updateOptions({ readOnly: true });
    setErrorMarker(null);
  } else editor.readOnly = true;
  setCode(info.binary ? `(fichier binaire, ${fmtN(info.size)} octets)` : info.text);
  if (monacoRef) editor.setScrollTop(0);
  $("error").hidden = true;
  drawTabs();
  drawTree();
  setStatus(`${node.path} · lecture seule`, "ok");
  renderFileLens(proj.open);
}

function closeFile() {
  if (!window.projectMode) return;
  window.projectMode = false;
  proj.open = null;
  if (monacoRef) { monacoRef.editor.setModelLanguage(editor.getModel(), "python"); editor.updateOptions({ readOnly: false }); }
  else editor.readOnly = false;
  setCode(proj.example);
  drawTabs();
  drawTree();
  ["input-card", "graph-card", "blocks-card", "depth-card", "train-card", "dream-card", "attack-card", "landscape-card"].forEach((id) => { const c = $(id); if (c) c.hidden = true; });
  run();
}

function drawTabs() {
  const bar = $("tabs");
  bar.hidden = !window.projectMode;
  bar.innerHTML = "";
  const tab = (label, on, click, icon) => {
    const b = document.createElement("button");
    b.className = `tab${on ? " on" : ""}`;
    b.innerHTML = `<span></span><b></b>`;
    b.querySelector("span").textContent = icon;
    b.querySelector("b").textContent = label;
    b.addEventListener("click", click);
    bar.appendChild(b);
  };
  tab(proj.exampleName, false, closeFile, "✎");
  if (proj.open) tab(proj.open.path, true, () => {}, iconOf(proj.open.name));
  const close = document.createElement("button");
  close.className = "tab close";
  close.textContent = "×";
  close.title = "Fermer le fichier, revenir à l'exemple";
  close.addEventListener("click", closeFile);
  bar.appendChild(close);
}

// ---------- À droite : ce que ce fichier est, et comment il tient aux autres ----------

function renderFileLens(f) {
  ["input-card", "graph-card", "blocks-card", "depth-card", "train-card", "dream-card", "attack-card", "landscape-card"].forEach((id) => { const c = $(id); if (c) c.hidden = true; });
  $("prediction").innerHTML = "";
  const root = $("layers");
  root.innerHTML = "";
  const folder = f.path.includes("/") ? f.path.split("/")[0] : null;
  const dir = folder && proj.data.tree.find((n) => n.path === folder);

  const card = mlCard("embed", f.name, f.lines ? `${f.lines} lignes` : f.binary ? `${fmtN(f.size)} octets` : "");
  addText(card, "explain", f.role);
  if (dir) addText(card, "meta", `Dossier ${folder}/ : ${dir.role}${dir.why ? " " + dir.why : ""}`);
  root.appendChild(card);

  if (f.symbols && f.symbols.length) {
    const plan = mlCard("dense", "Plan du fichier", `${f.symbols.length} définition${f.symbols.length > 1 ? "s" : ""}`);
    addText(plan, "explain", "Les fonctions et classes de ce fichier, dans l'ordre. Clique pour y aller dans le code : c'est la vue « Outline » de VS Code.");
    const list = document.createElement("div");
    list.className = "outline";
    f.symbols.forEach((s) => {
      const row = document.createElement("div");
      row.innerHTML = `<i></i><b></b><span></span>`;
      row.querySelector("i").textContent = s.kind === "classe" ? "C" : "ƒ";
      row.querySelector("b").textContent = s.name;
      row.querySelector("span").textContent = s.doc || `ligne ${s.line}`;
      row.addEventListener("click", () => revealLine(s.line));
      list.appendChild(row);
    });
    plan.appendChild(list);
    root.appendChild(plan);
  }

  const g = proj.data.graph;
  if (g.nodes.length && proj.root === "app") {
    const me = g.nodes.find((n) => n.path === f.path);
    const uses = me ? g.edges.filter((e) => e.source === me.id).map((e) => e.target) : [];
    const usedBy = me ? g.edges.filter((e) => e.target === me.id).map((e) => e.source) : [];
    const deps = mlCard("attn", "Dépendances du serveur", `${g.nodes.filter((n) => !n.test).length} modules`);
    addText(deps, "explain", me
      ? `Une flèche A → B veut dire « A importe B ». ${f.name} importe ${uses.length ? uses.join(", ") : "aucun module interne"} ; ${usedBy.length ? `il est importé par ${usedBy.join(", ")}` : "aucun module ne l'importe"}. Les modules du bas ne dépendent de rien : ce sont les briques ; ceux du haut assemblent.`
      : "Une flèche A → B veut dire « A importe B ». Les modules du bas ne dépendent de rien : ce sont les briques ; ceux du haut assemblent. Clique sur un module pour l'ouvrir.");
    deps.appendChild(moduleGraph(g, me ? me.id : null));
    root.appendChild(deps);
  }
}

// Les modules en étages : un module est au-dessus de ceux qu'il importe.
function moduleGraph(g, current) {
  const out = new Map(g.nodes.map((n) => [n.id, g.edges.filter((e) => e.source === n.id).map((e) => e.target)]));
  const rank = new Map();
  const depth = (id, seen = new Set()) => {
    if (rank.has(id)) return rank.get(id);
    if (seen.has(id)) return 0;
    seen.add(id);
    const r = out.get(id).length ? 1 + Math.max(...out.get(id).map((d) => depth(d, seen))) : 0;
    rank.set(id, r);
    return r;
  };
  g.nodes.forEach((n) => depth(n.id));
  const top = Math.max(...rank.values());
  g.nodes.filter((n) => n.test).forEach((n) => rank.set(n.id, top + 1));
  const rows = [];
  g.nodes.forEach((n) => (rows[rank.get(n.id)] ??= []).push(n));
  const W = 150, H = 36, gapX = 18, gapY = 46, pad = 16;
  const widest = Math.max(...rows.map((r) => r.length));
  const innerW = widest * W + (widest - 1) * gapX;
  const pos = new Map();
  rows.forEach((row, r) => row.forEach((n, k) => {
    const rowW = row.length * W + (row.length - 1) * gapX;
    pos.set(n.id, { x: pad + (innerW - rowW) / 2 + k * (W + gapX), y: pad + (rows.length - 1 - r) * (H + gapY) });
  }));
  const totalW = innerW + pad * 2, totalH = pad * 2 + rows.length * H + (rows.length - 1) * gapY;
  const svg = el("svg", { viewBox: `0 0 ${totalW} ${totalH}`, width: totalW, height: totalH, class: "agent-svg" });
  svg.appendChild(el("defs", {})).innerHTML = ["", "-on"].map((s) =>
    `<marker id="nl-parrow${s}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="agent-arrow${s}"/></marker>`).join("");
  g.edges.forEach((e) => {
    const s = pos.get(e.source), t = pos.get(e.target);
    if (!s || !t) return;
    const on = current && (e.source === current || e.target === current);
    const sx = s.x + W / 2, sy = s.y + H, tx = t.x + W / 2, ty = t.y - 1;
    svg.appendChild(el("path", { d: `M${sx},${sy} C${sx},${sy + 24} ${tx},${ty - 24} ${tx},${ty}`, class: `agent-edge${on ? " on" : ""}`, "marker-end": `url(#nl-parrow${on ? "-on" : ""})` }));
  });
  g.nodes.forEach((n) => {
    const p = pos.get(n.id);
    const node = svg.appendChild(el("g", { class: `agent-node step${n.id === current ? " active" : ""}${n.test ? " test" : ""}` }));
    node.appendChild(el("rect", { x: p.x, y: p.y, width: W, height: H, rx: 6 }));
    node.appendChild(el("text", { x: p.x + W / 2, y: p.y + 15, class: "blocks-label" }, n.id.replace("tests/", "")));
    node.appendChild(el("text", { x: p.x + W / 2, y: p.y + 29, class: "proj-lines" }, `${n.lines} lignes`));
    node.appendChild(el("title", {}, n.role));
    node.addEventListener("click", () => openFile({ name: n.path.split("/").pop(), path: n.path, kind: "file" }));
  });
  const box = document.createElement("div");
  box.className = "graph";
  box.appendChild(svg);
  return box;
}

(() => {
  $("project-btn").addEventListener("click", toggleExplorer);
  $("project-root").addEventListener("change", (e) => { loadProject(e.target.value); });
})();
