// neuro-lens : modèles scikit-learn — les données, la frontière de décision, ce que le modèle a appris.

const CLASS_COLORS = ["#5b8cff", "#ff9a85", "#2fbfa5", "#f2b84b", "#b083f0", "#4cd98a", "#ff6b6b", "#7d8597"];
const classColor = (i) => CLASS_COLORS[i % CLASS_COLORS.length];

const ML_EXPLAIN = {
  DecisionTreeClassifier: "Pose une suite de questions oui/non sur les colonnes ; chaque feuille de l'arbre donne une classe.",
  DecisionTreeRegressor: "Pose une suite de questions oui/non sur les colonnes ; chaque feuille de l'arbre donne une valeur.",
  RandomForestClassifier: "Fait voter beaucoup d'arbres de décision, chacun entraîné sur un échantillon différent des données.",
  RandomForestRegressor: "Fait la moyenne de beaucoup d'arbres de décision, chacun entraîné sur un échantillon différent des données.",
  GradientBoostingClassifier: "Empile de petits arbres : chacun corrige les erreurs des précédents.",
  LogisticRegression: "Trace une frontière droite : un poids par colonne, puis une sigmoïde transforme le score en probabilité.",
  LinearRegression: "Cherche la droite (ou le plan) qui passe au plus près des points : un poids par colonne.",
  Ridge: "Régression linéaire qui garde des poids petits pour moins coller au bruit.",
  KNeighborsClassifier: "Ne retient aucune règle : pour un nouveau point, regarde ses plus proches voisins et prend la classe majoritaire.",
  SVC: "Cherche la frontière qui laisse la plus grande marge entre les classes.",
  GaussianNB: "Suppose que chaque classe forme un nuage en cloche et choisit le nuage le plus probable.",
  KMeans: "Sans réponses attendues : place des centres et rattache chaque point au centre le plus proche.",
  DBSCAN: "Sans réponses attendues : regroupe les points serrés les uns contre les autres ; les isolés sont du bruit.",
  MLPClassifier: "Un petit réseau de neurones entraîné par scikit-learn.",
};
const TASKS = { classification: "classification", regression: "régression", clustering: "regroupement" };

function mlCard(family, title, right) {
  const c = document.createElement("div");
  c.className = "card";
  c.dataset.family = family;
  c.innerHTML = `<div class="card-head"><span class="tag"></span><span class="shape"></span></div>`;
  c.querySelector(".tag").textContent = title;
  c.querySelector(".shape").textContent = right ?? "";
  if (!right) c.querySelector(".shape").remove();
  return c;
}

function addText(parent, cls, text) {
  const p = document.createElement("p");
  p.className = cls;
  p.textContent = text;
  parent.appendChild(p);
  return p;
}

// Le réglage principal du modèle au curseur, avec la courbe qui montre le sur-apprentissage.
function knobCard(res, onKnob) {
  const k = res.knob, W = 420, H = 190, padL = 34, padB = 22, padT = 10;
  const card = mlCard("act", `Réglage : ${k.name}`, k.value === null ? "sans limite" : `${k.value}`);
  const supervised = Boolean(k.train);
  addText(card, "explain", supervised
    ? `Déplace le curseur : le modèle est réentraîné avec cette valeur, sans toucher au code. La courbe bleue est le score sur les exemples appris, l'orange sur ${k.held_out} exemples mis de côté. Quand la bleue continue de monter et que l'orange redescend, le modèle apprend par cœur : c'est le sur-apprentissage.`
    : k.inertia ? "Déplace le curseur pour changer le nombre de groupes. La courbe montre à quel point les points sont serrés autour de leur centre : là où elle fait un coude, ajouter des groupes n'apporte plus grand-chose."
    : "Déplace le curseur : le modèle est réentraîné avec cette valeur, sans toucher au code.");
  const at = k.values.indexOf(k.value);
  const slider = document.createElement("input");
  slider.type = "range"; slider.min = 0; slider.max = k.values.length - 1; slider.step = 1;
  slider.value = at >= 0 ? at : k.values.length - 1;
  slider.className = "knob";
  slider.setAttribute("aria-label", k.name);
  const label = addText(card, "meta", "");
  const say = () => { label.textContent = `${k.name} = ${k.values[slider.value]}` + (at < 0 ? ` (dans le code : ${k.value === null ? "sans limite" : k.value})` : ""); };
  say();
  slider.addEventListener("input", say);
  slider.addEventListener("change", () => { window.knobHadFocus = true; onKnob(k.name, k.values[slider.value]); });
  card.insertBefore(slider, label);
  if (window.knobHadFocus) { window.knobHadFocus = false; setTimeout(() => slider.focus(), 0); }

  const series = supervised ? [[k.train, classColor(0), "exemples appris"], [k.test, classColor(3), "exemples mis de côté"]]
    : k.inertia ? [[k.inertia, classColor(0), "dispersion (inertie)"]] : [];
  if (!series.length) return card;
  const [cv, ctx] = plotArea(W, H);
  const all = series.flatMap((s) => s[0]);
  const lo = supervised ? Math.min(0.5, ...all) : 0, hi = supervised ? 1 : Math.max(...all);
  const x = (i) => padL + (i / Math.max(1, k.values.length - 1)) * (W - padL - 12);
  const y = (v) => H - padB - ((v - lo) / (hi - lo || 1)) * (H - padB - padT);
  ctx.font = "10px system-ui"; ctx.fillStyle = "#8b93a7"; ctx.strokeStyle = "#2a3142";
  [lo, (lo + hi) / 2, hi].forEach((v) => {
    ctx.beginPath(); ctx.moveTo(padL, y(v)); ctx.lineTo(W - 12, y(v)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(supervised ? `${Math.round(v * 100)} %` : v.toFixed(0), padL - 4, y(v) + 3);
  });
  ctx.textAlign = "center";
  k.values.forEach((v, i) => ctx.fillText(v, x(i), H - 6));
  if (at >= 0) { // la valeur en cours
    ctx.strokeStyle = "#e6e8ee"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x(at), padT); ctx.lineTo(x(at), H - padB); ctx.stroke(); ctx.setLineDash([]);
  }
  series.forEach(([values, color]) => {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
    values.forEach((v, i) => { ctx.beginPath(); ctx.arc(x(i), y(v), i === at ? 4.5 : 2.5, 0, 2 * Math.PI); ctx.fill(); });
  });
  card.appendChild(cv);
  card.appendChild(legendOf(series.map((s) => [s[2], s[1]])));
  if (supervised) {
    const best = k.test.indexOf(Math.max(...k.test));
    addText(card, "meta", `Meilleur score sur les exemples mis de côté : ${(k.test[best] * 100).toFixed(1)} % avec ${k.name} = ${k.values[best]}` +
      (at >= 0 ? ` · ici : ${(k.test[at] * 100).toFixed(1)} %` : ""));
  }
  return card;
}

function legendOf(pairs) {
  const box = document.createElement("div");
  box.className = "legend";
  pairs.forEach(([name, color]) => {
    const s = document.createElement("span");
    s.innerHTML = `<i style="background:${color}"></i>`;
    s.appendChild(document.createTextNode(name));
    box.appendChild(s);
  });
  return box;
}

function renderML(root, res, onHover, onPick, onKnob) {
  root.innerHTML = "";
  const score = res.score === undefined ? null
    : res.task === "regression" ? `R² ${res.score.toFixed(3)}` : `${(res.score * 100).toFixed(1)} % de bonnes réponses`;

  const head = mlCard("dense", res.estimator, score);
  head.dataset.line = res.line ?? "";
  if (ML_EXPLAIN[res.estimator]) addText(head, "explain", ML_EXPLAIN[res.estimator]);
  const params = Object.entries(res.params).map(([k, v]) => `${k}=${v}`).join(", ");
  addText(head, "meta", [
    `${TASKS[res.task]} · ${res.n_samples} exemples · ${res.features.length} colonne${res.features.length > 1 ? "s" : ""} (${res.features.slice(0, 6).join(", ")}${res.features.length > 6 ? "…" : ""})`,
    res.steps.length ? `chaîne : ${res.steps.join(" → ")}` : "",
    res.trained_here ? "entraîné ici avec model.fit" : "déjà entraîné dans ton code",
  ].filter(Boolean).join(" · "));
  if (score) addText(head, "meta", "Le score est mesuré sur les données d'entraînement : il est optimiste. Garde des données de côté (train_test_split) pour une vraie mesure.");
  if (params) addText(head, "meta", params).classList.add("mono");
  head.addEventListener("mouseenter", () => onHover(res.line));
  head.addEventListener("mouseleave", () => onHover(null));
  head.addEventListener("click", () => onPick(res.line));
  root.appendChild(head);
  if (res.knob && onKnob) root.appendChild(knobCard(res, onKnob));

  if (res.scatter) root.appendChild(scatterCard(res));
  if (res.curve) root.appendChild(curveCard(res));
  if (res.fit) root.appendChild(fitCard(res));
  if (res.tree) root.appendChild(treeCard(res));
  if (res.importances) root.appendChild(barsCard("Importance des colonnes", res.importances,
    "Part de chaque colonne dans les décisions du modèle (total 100 %).", true));
  if (res.coefficients) root.appendChild(barsCard("Poids appris", res.coefficients,
    "Un poids par colonne : positif (bleu) pousse la réponse vers le haut, négatif (rouge) vers le bas." +
    (res.coefficients_of ? ` Poids de la classe « ${res.coefficients_of} ».` : ""), false));
}

// ---------- Tracés ----------

function plotArea(w, h) {
  const cv = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = `${w}px`; cv.style.height = `${h}px`;
  cv.className = "plot";
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  return [cv, ctx];
}

function scales(xs, ys, w, h, pad, range) {
  const span = (v) => { const lo = Math.min(...v), hi = Math.max(...v); return hi > lo ? [lo, hi] : [lo - 1, hi + 1]; };
  const [x0, x1] = range?.x ?? span(xs), [y0, y1] = range?.y ?? span(ys);
  return [(v) => pad + ((v - x0) / (x1 - x0)) * (w - 2 * pad), (v) => h - pad - ((v - y0) / (y1 - y0)) * (h - 2 * pad)];
}

function legend(names) {
  const box = document.createElement("div");
  box.className = "legend";
  names.forEach((n, i) => {
    const s = document.createElement("span");
    s.innerHTML = `<i style="background:${classColor(i)}"></i>`;
    s.appendChild(document.createTextNode(n));
    box.appendChild(s);
  });
  return box;
}

function scatterCard(res) {
  const s = res.scatter, W = 420, H = 320, pad = s.regions ? 0 : 16;
  const card = mlCard("conv", s.regions ? "Données et frontière de décision" : "Données", s.axes.filter(Boolean).join(" × "));
  addText(card, "explain", s.regions
    ? "Chaque point est un exemple, coloré par la réponse du modèle. Le fond montre ce que le modèle répondrait partout ailleurs : là où la couleur change, c'est sa frontière."
    : s.projected ? "Chaque point est un exemple, coloré par la réponse du modèle. Il y a plus de 2 colonnes : elles sont ramenées à 2 axes (ACP) pour tenir sur un plan."
    : "Chaque point est un exemple, coloré par la réponse du modèle.");
  const [cv, ctx] = plotArea(W, H);
  const [sx, sy] = scales(s.points.map((p) => p[0]), s.points.map((p) => p[1]), W, H, pad, s.regions);
  if (s.regions) {
    const n = s.regions.cells.length, cw = W / n, ch = H / n;
    s.regions.cells.forEach((row, r) => row.forEach((c, k) => {
      ctx.fillStyle = classColor(c) + "40";
      const x0 = Math.round(k * cw), y0 = Math.round(H - (r + 1) * ch); // bords entiers : pas de lignes entre les cases
      ctx.fillRect(x0, y0, Math.round((k + 1) * cw) - x0, Math.round(H - r * ch) - y0);
    }));
  }
  s.points.forEach(([x, y], i) => {
    ctx.beginPath();
    ctx.arc(sx(x), sy(y), 3.2, 0, 2 * Math.PI);
    ctx.fillStyle = classColor(s.labels[i]);
    ctx.fill();
    if (s.wrong?.[i]) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
  });
  card.appendChild(cv);
  card.appendChild(legend(res.classes));
  const wrong = s.wrong?.filter(Boolean).length;
  if (wrong) addText(card, "meta", `${wrong} point${wrong > 1 ? "s" : ""} cerclé${wrong > 1 ? "s" : ""} de blanc : le modèle se trompe dessus.`);
  return card;
}

function curveCard(res) {
  const c = res.curve, W = 420, H = 300;
  const card = mlCard("conv", "Données et courbe apprise", c.axes.join(" → "));
  addText(card, "explain", "Les points sont les exemples ; la ligne est ce que le modèle prédit pour chaque valeur.");
  const [cv, ctx] = plotArea(W, H);
  const [sx, sy] = scales(c.points.map((p) => p[0]), c.points.map((p) => p[1]).concat(c.y), W, H, 16);
  ctx.fillStyle = classColor(0);
  c.points.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(sx(x), sy(y), 3, 0, 2 * Math.PI); ctx.fill(); });
  ctx.strokeStyle = classColor(3); ctx.lineWidth = 2; ctx.beginPath();
  c.x.forEach((x, i) => (i ? ctx.lineTo(sx(x), sy(c.y[i])) : ctx.moveTo(sx(x), sy(c.y[i]))));
  ctx.stroke();
  card.appendChild(cv);
  return card;
}

function fitCard(res) {
  const pts = res.fit.points, W = 320, H = 320;
  const card = mlCard("conv", "Prédit contre attendu");
  addText(card, "explain", "Un point par exemple : la vraie valeur en horizontal, la prédiction en vertical. Sur la diagonale, le modèle a tout juste.");
  const [cv, ctx] = plotArea(W, H);
  const all = pts.flat(), range = { x: [Math.min(...all), Math.max(...all)], y: [Math.min(...all), Math.max(...all)] };
  const [sx, sy] = scales(all, all, W, H, 16, range);
  ctx.strokeStyle = "#8b93a7"; ctx.setLineDash([4, 4]); ctx.beginPath();
  ctx.moveTo(sx(range.x[0]), sy(range.x[0])); ctx.lineTo(sx(range.x[1]), sy(range.x[1])); ctx.stroke();
  ctx.fillStyle = classColor(0);
  pts.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(sx(x), sy(y), 3, 0, 2 * Math.PI); ctx.fill(); });
  card.appendChild(cv);
  return card;
}

// ---------- Arbre de décision ----------

function treeCard(res) {
  const t = res.tree, bw = 112, bh = 40, gx = 10, gy = 34;
  const card = mlCard("pool", "Arbre de décision", `${t.leaves} feuilles · profondeur ${t.depth}`);
  addText(card, "explain", "On part du haut. À chaque boîte, une question : si oui on descend à gauche, sinon à droite. En bas, la réponse du modèle.");
  let next = 0, deepest = 0;
  (function place(n, d) {
    n.d = d; deepest = Math.max(deepest, d);
    if (n.children) { n.children.forEach((c) => place(c, d + 1)); n.x = (n.children[0].x + n.children[1].x) / 2; }
    else n.x = next++ * (bw + gx);
  })(t.root, 0);
  const W = next * (bw + gx) - gx + 8, H = (deepest + 1) * (bh + gy) - gy + 8;
  const svg = el("svg", { viewBox: `-4 -4 ${W} ${H}`, width: W, height: H, class: "tree-svg" });
  (function draw(n) {
    const y = n.d * (bh + gy);
    (n.children ?? []).forEach((c, k) => {
      const cy = c.d * (bh + gy);
      svg.appendChild(el("line", { x1: n.x + bw / 2, y1: y + bh, x2: c.x + bw / 2, y2: cy, class: "tree-edge" }));
      svg.appendChild(el("text", { x: (n.x + c.x) / 2 + bw / 2, y: (y + bh + cy) / 2 + 3, class: "tree-yesno" }, k ? "non" : "oui"));
      draw(c);
    });
    const g = svg.appendChild(el("g", { class: n.children ? "tree-node" : "tree-node leaf" }));
    const rect = g.appendChild(el("rect", { x: n.x, y, width: bw, height: bh, rx: 6 }));
    if (n.cls !== undefined) { rect.style.stroke = classColor(n.cls); if (!n.children) rect.style.fill = classColor(n.cls) + "33"; }
    g.appendChild(el("text", { x: n.x + bw / 2, y: y + 16, class: "blocks-label" },
      n.children ? `${n.feature} ≤ ${n.threshold} ?` : n.cut ? `${n.label} …` : n.label));
    g.appendChild(el("text", { x: n.x + bw / 2, y: y + 30, class: "blocks-sub" },
      `${n.samples} ex.${n.purity !== undefined ? ` · ${Math.round(n.purity * 100)} % ${n.children ? n.label : "purs"}` : n.children ? ` · ${n.label}` : ""}`));
  })(t.root);
  const box = document.createElement("div");
  box.className = "graph";
  box.appendChild(svg);
  card.appendChild(box);
  if (res.n_trees) addText(card, "meta", `Le modèle fait voter ${res.n_trees} arbres : celui-ci est le premier.`);
  if (t.depth > deepest) addText(card, "meta", `L'arbre descend jusqu'à ${t.depth} niveaux ; « … » marque les branches coupées à l'affichage.`);
  return card;
}

function barsCard(title, bars, explain, share) {
  const card = mlCard("dense", title);
  addText(card, "explain", explain);
  const max = Math.max(...bars.map((b) => Math.abs(b.value))) || 1;
  const box = document.createElement("div");
  box.className = "hbars";
  bars.forEach((b) => {
    const row = document.createElement("div");
    row.innerHTML = `<span></span><div><i></i></div><b></b>`;
    row.querySelector("span").textContent = b.name;
    row.querySelector("i").style.width = `${(Math.abs(b.value) / max) * 100}%`;
    row.querySelector("i").style.background = b.value >= 0 ? "var(--conv)" : "var(--danger)";
    row.querySelector("b").textContent = share ? `${(b.value * 100).toFixed(0)} %` : b.value.toFixed(3);
    box.appendChild(row);
  });
  card.appendChild(box);
  return card;
}
