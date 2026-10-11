// neuro-lens : la carte 3D de la chaîne, de la donnée à la production.
//
// Quatre plateformes — Données, Modèle, Infrastructure, Production — et sur chacune les briques du
// projet. La 3D porte une information : une brique qui existe vraiment se dresse en volume, une brique
// prête mais pas branchée reste basse, une brique absente est à plat. Un clic l'explique simplement :
// à quoi elle sert, ce qu'elle vaut dans ce projet, ce qui se passe sans elle, où la trouver.

const CHAIN = { scale: 44, pitch: 0.62, yawMin: -1.15, yawMax: -0.2, blockW: 2.3, blockD: 0.78, rowGap: 1.5, stageW: 3.1, stageGap: 0.75 };
const HEIGHT = { built: 0.95, ready: 0.34, missing: 0.03 };
const STATE_TEXT = { built: "en place", ready: "prête, pas encore branchée", missing: "pas encore là" };
const STAGE_COLOR = ["#ff9a85", "#b083f0", "#5b8cff", "#4cd98a"];

let chainYaw = -0.62;
let chainPicked = null;   // identifiant de la brique expliquée
let chainModel = null;
let chainOpen = (() => { try { return localStorage.getItem("neuro-lens-chaine") !== "0"; } catch (e) { return true; } })();

const brick = (id, label, state, value, role, without, where = {}) => ({ id, label, state, value, role, without, ...where });
const plural = (n, word) => `${n} ${word}${n > 1 ? "s" : ""}`;

// ---------- Le modèle de la chaîne, selon le type de projet ----------

function chainForML(res) {
  const ops = res.mlops || {}, prod = ops.production, mon = ops.monitoring, knob = res.knob || {};
  const infra = typeof opsInfra !== "undefined" && opsInfra ? opsInfra : { phase: "idle" };
  const imageBuilt = ["starting", "running", "stopped"].includes(infra.phase);
  const at = knob.values ? knob.values.indexOf(knob.value) : -1;
  const held = knob.test && at >= 0 ? knob.test[at] : null;
  return {
    title: "Projet de ML classique",
    stages: [
      ["Données", [
        brick("source", "Jeu de données", "built", `${plural(res.n_samples, "exemple")}, ${plural(res.features.length, "colonne")}`,
          "La matière première : des exemples décrits par des colonnes, avec pour chacun la réponse attendue.",
          "Sans données fiables, le meilleur modèle apprend n'importe quoi.", { file: "src/train.py" }),
        brick("split", "Mise de côté", knob.held_out ? "built" : "missing", knob.held_out ? `${knob.held_out} exemples cachés au modèle` : "aucune",
          "Une partie des exemples est cachée pendant l'apprentissage, pour juger le modèle honnêtement ensuite.",
          "Sans elle, le score mesure ce que le modèle a appris par cœur, pas ce qu'il sait faire."),
        brick("print", "Empreinte", "built", res.data_id,
          "Une signature courte des données : si elle change, ce ne sont plus les mêmes données.",
          "Impossible de dire sur quelles données une version a été entraînée.", { target: "ops-runs" }),
      ]],
      ["Modèle", [
        brick("train", "Entraînement", "built", res.estimator + (knob.name ? ` · ${knob.name}=${knob.value}` : ""),
          "Le modèle cherche, dans les exemples, la règle qui relie les colonnes à la réponse.",
          "Pas de modèle : seulement des données.", { file: "src/train.py" }),
        brick("eval", "Évaluation", held !== null ? "built" : res.score !== undefined ? "ready" : "missing",
          held !== null ? `${(held * 100).toFixed(1)} % sur les exemples cachés` : res.score !== undefined ? `${(res.score * 100).toFixed(1)} % sur les exemples appris seulement` : "aucune",
          "On mesure le modèle sur des exemples qu'il n'a jamais vus : c'est le seul score qui compte.",
          "Un modèle non mesuré part en production à l'aveugle."),
        brick("runs", "Suivi des essais", (ops.runs || []).length ? "built" : "missing", plural((ops.runs || []).length, "essai") + " noté" + ((ops.runs || []).length > 1 ? "s" : ""),
          "Chaque combinaison modèle, réglage et données essayée est notée avec ses scores.",
          "Dans un mois, personne ne saura pourquoi cette version a été choisie.", { target: "ops-runs" }),
      ]],
      ["Infrastructure", [
        brick("tests", "Tests", "built", "générés, à lancer",
          "Des vérifications automatiques : le script tourne, et le modèle reste au-dessus d'un seuil de qualité.",
          "Un changement qui casse tout passe inaperçu jusqu'en production.", { file: "tests/test_train.py" }),
        brick("version", "Version figée", prod ? "built" : "missing", prod ? `v${prod.version} au registre` : "rien de déployé",
          "Le modèle entraîné est gelé sous un numéro, avec sa fiche : il ne changera plus jamais.",
          "Impossible de comparer deux modèles ou de revenir en arrière.", { target: "ops-deploy" }),
        brick("image", "Image Docker", imageBuilt ? "built" : prod ? "ready" : "missing", imageBuilt ? `${infra.size_mb ?? "?"} Mo` : prod ? "Dockerfile prêt, image pas construite" : "pas de Dockerfile",
          "Une boîte fermée qui contient Python, les bibliothèques et le modèle : identique partout où elle tourne.",
          "« Ça marche sur ma machine » : le service se comporte autrement ailleurs.", { target: "ops-infra" }),
        brick("ci", "Intégration continue", "ready", "fichier prêt, pas branché sur GitHub",
          "À chaque modification poussée, une machine neuve installe le projet et lance les tests.",
          "Les tests ne sont lancés que quand quelqu'un y pense.", { file: ".github/workflows/tests.yml" }),
      ]],
      ["Production", [
        brick("api", "API de prédiction", prod ? "built" : "missing", prod ? `v${prod.version} répond` : "rien en service",
          "La porte d'entrée : une application envoie des valeurs, le modèle renvoie sa réponse.",
          "Le modèle reste dans un fichier : personne ne peut s'en servir.", { target: "ops-api" }),
        brick("box", "Conteneur", infra.phase === "running" ? "built" : imageBuilt ? "ready" : "missing", infra.phase === "running" ? `en marche, port ${infra.port}` : imageBuilt ? "image prête, conteneur arrêté" : "pas lancé",
          "L'image, une fois lancée : le service tourne, isolé, et répond sur un port.",
          "Le service dépend de la machine où on l'a installé à la main.", { target: "ops-infra" }),
        brick("watch", "Surveillance", !mon ? "missing" : mon.status === "few" ? "ready" : "built",
          !mon ? "aucun trafic observé" : mon.status === "few" ? `${plural(mon.requests, "requête")}, trop peu pour juger` : mon.status === "alert" ? "alerte : les données ont changé" : mon.status === "watch" ? "à surveiller" : `${plural(mon.requests, "requête")}, tout va bien`,
          "On compare ce que le modèle reçoit en production à ce qu'il a appris : latence, qualité, dérive.",
          "Le modèle devient mauvais sans que personne le voie : il continue de répondre, mais faux.", { target: "ops-watch" }),
        brick("retrain", "Réentraînement", "missing", "pas encore dans l'outil",
          "Quand les données ont changé, on réentraîne sur des données récentes et on redéploie : la boucle se referme.",
          "Chaque dérive demande une intervention à la main, souvent trop tard."),
      ]],
    ],
    loop: ["watch", "source"],
  };
}

function chainForTorch(res) {
  const text = Boolean(res.input.tokens), t = res.training, epochs = t ? t.epochs.length : 0;
  const last = epochs ? t.epochs[epochs - 1] : null;
  const none = (label, role, without) => brick(label.toLowerCase().replace(/\W+/g, "-"), label, "missing", "pas encore pour les réseaux", role, without);
  return {
    title: text ? "Projet de réseau de neurones (texte)" : "Projet de réseau de neurones (image)",
    stages: [
      ["Données", [
        brick("input", "Entrée", "built", text ? `${plural(res.input.tokens.length, "token")}` : `image ${res.input.shape.slice(1).join(" × ")}`,
          text ? "La phrase, découpée en tokens : chaque mot reçoit un numéro." : "L'image que tu dessines : une grille de nombres entre 0 et 1.",
          "Rien n'entre dans le réseau : rien n'en sort.", { target: "input-card" }),
        brick("dataset", "Jeu d'entraînement", epochs ? "built" : res.trainable ? "ready" : "missing",
          epochs ? "1 437 chiffres manuscrits" : res.trainable ? "1 437 chiffres prêts, pas encore utilisés" : "aucun pour ce modèle",
          "Les exemples sur lesquels le réseau apprend, avec la bonne réponse pour chacun.",
          "Le réseau garde ses poids tirés au hasard : ses réponses le sont aussi.", { target: "train-card" }),
      ]],
      ["Modèle", [
        brick("arch", "Architecture", "built", `${plural(res.layers.length, "couche")}, ${fmtN(res.n_params)} paramètres`,
          "L'empilement de couches que tu écris : c'est lui qui transforme l'entrée en réponse.",
          "Pas de réseau.", { file: "src/model.py", target: "graph-card" }),
        brick("fit", "Entraînement", epochs ? "built" : res.trainable ? "ready" : "missing", epochs ? plural(epochs, "époque") : res.trainable ? "prêt, pas lancé" : "pas disponible ici",
          "Le réseau ajuste ses poids, exemple après exemple, pour se tromper de moins en moins.",
          "Les poids restent ceux du hasard.", { target: "train-card" }),
        brick("score", "Évaluation", last ? "built" : "missing", last ? `${Math.round(last.test_acc * 100)} % sur des chiffres jamais vus` : "aucune mesure",
          "On mesure le réseau sur des exemples qu'il n'a jamais vus.",
          "On ne sait pas s'il a appris ou seulement mémorisé.", { target: "train-card" }),
      ]],
      ["Infrastructure", [
        brick("tests", "Tests", "built", "générés, à lancer",
          "Des vérifications automatiques : la forme de sortie et le nombre de paramètres ne changent pas par accident.",
          "Une modification casse le réseau sans que personne le remarque.", { file: "tests/test_model.py" }),
        brick("ci", "Intégration continue", "ready", "fichier prêt, pas branché sur GitHub",
          "À chaque modification poussée, une machine neuve installe le projet et lance les tests.",
          "Les tests ne sont lancés que quand quelqu'un y pense.", { file: ".github/workflows/tests.yml" }),
        none("Version figée", "Les poids appris, gelés sous un numéro avec leur fiche.", "Un entraînement réussi est perdu au redémarrage."),
        none("Image Docker", "Une boîte fermée avec Python, PyTorch et les poids : identique partout.", "Le réseau ne tourne que sur la machine où il a été entraîné."),
      ]],
      ["Production", [
        none("API de prédiction", "La porte d'entrée : une application envoie une image, le réseau renvoie sa réponse.", "Le réseau ne sert à personne d'autre que toi."),
        none("Surveillance", "On vérifie que ce qu'il reçoit ressemble à ce qu'il a appris.", "Il devient mauvais sans que personne le voie."),
      ]],
    ],
    loop: null,
  };
}

function chainForAgent(res) {
  const ops = res.ops || {}, steps = res.nodes.filter((n) => n.kind === "step").length;
  const live = res.llm && res.llm.live, ev = ops.eval && ops.eval.variants ? ops.eval.variants : null;
  const rules = ops.rules || [], limits = ops.limits ? Object.entries(ops.limits).map(([k, v]) => `${v} ${k}`).join(", ") : "";
  return {
    title: "Projet d'agent",
    stages: [
      ["Données", [
        brick("ask", "Demande", res.start ? "built" : "missing", res.start ? `${plural(Object.keys(res.start).length, "champ")} dans l'état de départ` : "pas d'état de départ",
          "Ce qu'on confie à l'agent, et sa mémoire de travail : l'état qu'il fait évoluer d'étape en étape.",
          "L'agent n'a rien à faire.", { file: "src/agent.py" }),
        brick("tools", "Outils", steps > 1 ? "built" : "ready", plural(steps, "étape") + " dans le graphe",
          "Ce que l'agent peut faire au-delà de parler : chercher, calculer, appeler un service.",
          "L'agent ne peut que répondre de mémoire, donc inventer."),
      ]],
      ["Modèle", [
        brick("brain", "Cerveau", live ? "built" : "ready", live ? "un vrai modèle de langage" : "une règle fixe (pas de modèle appelé)",
          "Ce qui décide de la prochaine action. Ici une règle simple peut jouer ce rôle pour tester sans rien payer.",
          "Personne ne décide : l'agent déroule toujours le même chemin."),
        brick("graph", "Graphe d'étapes", "built", `${plural(steps, "étape")}, ${plural(res.edges.length, "chemin")}`,
          "Le plan des étapes et des chemins possibles entre elles, boucles comprises.",
          "Un agent sans plan visible : impossible de savoir par où il peut passer.", { file: "src/agent.py" }),
        brick("guard", "Garde-fous", rules.length ? "built" : "missing", rules.length ? plural(rules.length, "règle") + " vérifiée" + (rules.length > 1 ? "s" : "") : "aucun",
          "Des règles vérifiées sur la réponse avant de la montrer : longueur, mots interdits, données personnelles.",
          "Une réponse fausse, trop longue ou indiscrète part telle quelle à l'utilisateur."),
      ]],
      ["Infrastructure", [
        brick("tests", "Tests", res.start ? "built" : "missing", res.start ? "générés, à lancer" : "aucun",
          "Une vérification automatique : l'agent va au bout de sa tâche sans tourner en rond.",
          "Une boucle infinie n'est découverte qu'en production.", { file: "tests/test_agent.py" }),
        brick("budget", "Budget", limits ? "built" : "missing", limits ? `plafond : ${limits}` : "aucun plafond",
          "Un plafond d'appels et de tokens : au-delà, l'agent s'arrête.",
          "Un agent qui boucle coûte sans limite."),
        brick("ci", "Intégration continue", "ready", "fichier prêt, pas branché sur GitHub",
          "À chaque modification poussée, une machine neuve installe le projet et lance les tests.",
          "Les tests ne sont lancés que quand quelqu'un y pense.", { file: ".github/workflows/tests.yml" }),
      ]],
      ["Production", [
        brick("trace", "Trace", (ops.spans || []).length ? "built" : "missing", plural((ops.spans || []).length, "étape") + " enregistrée" + ((ops.spans || []).length > 1 ? "s" : ""),
          "Le journal de tout ce que l'agent a fait : chaque étape, chaque appel, sa durée.",
          "Quand il se trompe, impossible de savoir à quelle étape."),
        brick("cost", "Coût", ops.calls ? "built" : "ready", ops.calls ? `${plural(ops.calls, "appel")}, ${ops.tokens_in + ops.tokens_out} tokens, ${ops.cost} $` : "0 appel payant pour l'instant",
          "Ce que chaque tâche consomme : appels au modèle, tokens, argent.",
          "La facture arrive sans qu'on sache quelle tâche l'a causée."),
        brick("evals", "Évaluation", ev ? "built" : "missing", ev ? plural(ev.length, "variante") + " comparée" + (ev.length > 1 ? "s" : "") + ` sur ${plural(ev[0].rows.length, "cas")}` : "aucun cas de test",
          "Les mêmes questions rejouées à chaque changement, pour savoir si l'agent s'améliore ou régresse.",
          "On change le prompt « au feeling », sans savoir ce qu'on a cassé."),
      ]],
    ],
    loop: ["evals", "ask"],
  };
}

function chainFor(res) {
  const model = res.kind === "ml" ? chainForML(res) : res.kind === "agent" ? chainForAgent(res) : chainForTorch(res);
  model.stages = model.stages.map(([name, blocks]) => ({ name, blocks }));
  return model;
}

// ---------- Le dessin 3D ----------

function chainProject(x, y, z) {
  const c = Math.cos(chainYaw), s = Math.sin(chainYaw);
  const xr = x * c - y * s, yr = x * s + y * c;
  return { x: xr * CHAIN.scale, y: (yr * CHAIN.pitch - z) * CHAIN.scale * 0.92, depth: yr };
}

function shade(hex, k) { // k < 1 assombrit, k > 1 éclaircit
  const n = parseInt(hex.slice(1), 16);
  const ch = [n >> 16, (n >> 8) & 255, n & 255].map((v) => Math.max(0, Math.min(255, Math.round(k <= 1 ? v * k : v + (255 - v) * (k - 1)))));
  return `rgb(${ch.join(",")})`;
}

function drawChain() {
  const root = $("chain"), model = chainModel;
  if (!model) return;
  root.innerHTML = "";
  const g = CHAIN, items = [], boxes = [];
  const rows = Math.max(...model.stages.map((s) => s.blocks.length));
  const depth = rows * g.rowGap + 0.5;
  model.stages.forEach((stage, i) => {
    const x0 = i * (g.stageW + g.stageGap);
    items.push({ kind: "stage", i, name: stage.name, pts: [[x0, 0], [x0 + g.stageW, 0], [x0 + g.stageW, depth], [x0, depth]] });
    stage.blocks.forEach((b, k) => {
      const bx = x0 + (g.stageW - g.blockW) / 2, by = 0.45 + k * g.rowGap;
      boxes.push({ b, i, x: bx, y: by, w: g.blockW, d: g.blockD, h: HEIGHT[b.state] });
    });
  });
  const centre = (id) => { const o = boxes.find((v) => v.b.id === id); return o ? [o.x + o.w / 2, o.y + o.d / 2] : null; };

  // Tout projeter une première fois pour cadrer la scène.
  const all = [];
  items.forEach((s) => s.pts.forEach(([x, y]) => all.push(chainProject(x, y, 0))));
  boxes.forEach((o) => [[o.x, o.y], [o.x + o.w, o.y], [o.x + o.w, o.y + o.d], [o.x, o.y + o.d]].forEach(([x, y]) => { all.push(chainProject(x, y, 0)); all.push(chainProject(x, y, o.h + 0.55)); }));
  const minX = Math.min(...all.map((p) => p.x)) - 14, maxX = Math.max(...all.map((p) => p.x)) + 14;
  const minY = Math.min(...all.map((p) => p.y)) - 18, maxY = Math.max(...all.map((p) => p.y)) + 26;
  const P = (x, y, z) => { const p = chainProject(x, y, z); return [p.x - minX, p.y - minY]; };
  const poly = (pts) => pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ");
  const W = maxX - minX, H = maxY - minY;
  const svg = el("svg", { viewBox: `0 0 ${W.toFixed(0)} ${H.toFixed(0)}`, width: W.toFixed(0), height: H.toFixed(0), class: "chain-svg", role: "img", "aria-label": `Carte de la chaîne : ${model.title}` });

  // Les plateformes, de la plus lointaine à la plus proche.
  items.slice().sort((a, b) => chainProject(a.pts[0][0] + g.stageW / 2, depth / 2, 0).depth - chainProject(b.pts[0][0] + g.stageW / 2, depth / 2, 0).depth).forEach((s) => {
    svg.appendChild(el("polygon", { points: poly(s.pts.map(([x, y]) => P(x, y, 0))), class: "chain-stage", style: `stroke:${STAGE_COLOR[s.i]}` }));
    const corners = s.pts.map(([x, y]) => ({ x, y, p: chainProject(x, y, 0) }));
    const near = corners.slice().sort((a, b) => b.p.depth - a.p.depth).slice(0, 2); // le bord le plus proche de l'œil
    const [lx, ly] = P((near[0].x + near[1].x) / 2, (near[0].y + near[1].y) / 2, 0);
    svg.appendChild(el("text", { x: lx, y: ly + 15, class: "chain-stage-name", style: `fill:${STAGE_COLOR[s.i]}` }, `${s.i + 1}. ${s.name}`));
  });

  // Les flux : d'une brique à la suivante, puis d'une plateforme à la suivante ; la boucle de retour en pointillés.
  const flow = (a, b, cls) => {
    if (!a || !b) return;
    const [x1, y1] = P(a[0], a[1], 0), [x2, y2] = P(b[0], b[1], 0);
    svg.appendChild(el("line", { x1, y1, x2, y2, class: `chain-flow ${cls}` }));
  };
  model.stages.forEach((stage, i) => {
    stage.blocks.forEach((b, k) => { if (k) flow(centre(stage.blocks[k - 1].id), centre(b.id), ""); });
    if (i) flow(centre(model.stages[i - 1].blocks[model.stages[i - 1].blocks.length - 1].id), centre(stage.blocks[0].id), "");
  });
  if (model.loop) flow(centre(model.loop[0]), centre(model.loop[1]), "loop");

  // Les briques, de la plus lointaine à la plus proche ; dans chaque brique, les faces dans le même ordre.
  boxes.sort((a, b) => chainProject(a.x + a.w / 2, a.y + a.d / 2, 0).depth - chainProject(b.x + b.w / 2, b.y + b.d / 2, 0).depth).forEach((o) => {
    const color = STAGE_COLOR[o.i], b = o.b;
    const grp = svg.appendChild(el("g", { class: `chain-brick ${b.state}${b.id === chainPicked ? " picked" : ""}`, tabindex: 0, role: "button", "aria-label": `${b.label} : ${STATE_TEXT[b.state]}. ${b.value}` }));
    const c = [[o.x, o.y], [o.x + o.w, o.y], [o.x + o.w, o.y + o.d], [o.x, o.y + o.d]];
    const sides = [0, 1, 2, 3].map((k) => {
      const [a, d] = [c[k], c[(k + 1) % 4]];
      return { pts: [P(a[0], a[1], 0), P(d[0], d[1], 0), P(d[0], d[1], o.h), P(a[0], a[1], o.h)], depth: chainProject((a[0] + d[0]) / 2, (a[1] + d[1]) / 2, 0).depth, k };
    }).sort((p, q) => p.depth - q.depth);
    if (b.state !== "missing") sides.forEach((s) => grp.appendChild(el("polygon", { points: poly(s.pts), fill: shade(color, s.k % 2 ? 0.42 : 0.58), class: "chain-face" })));
    grp.appendChild(el("polygon", { points: poly(c.map(([x, y]) => P(x, y, o.h))), fill: b.state === "missing" ? "none" : shade(color, b.state === "built" ? 0.9 : 0.6), stroke: color, class: "chain-top" }));
    const [tx, ty] = P(o.x + o.w / 2, o.y + o.d / 2, o.h);
    grp.appendChild(el("text", { x: tx, y: ty - 9, class: "chain-label" }, b.label));
    grp.appendChild(el("title", {}, `${b.label} — ${STATE_TEXT[b.state]} · ${b.value}`));
    const pick = () => { chainPicked = b.id; drawChain(); };
    grp.addEventListener("click", pick);
    grp.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    svg.appendChild(grp);
  });
  root.appendChild(svg);
  drawChainInfo();
}

function drawChainInfo() {
  const info = $("chain-info"), model = chainModel;
  const all = model.stages.flatMap((s, i) => s.blocks.map((b) => ({ b, stage: s.name, i })));
  const built = all.filter((v) => v.b.state === "built").length;
  $("chain-shape").textContent = `${built} brique${built > 1 ? "s" : ""} en place sur ${all.length}`;
  $("chain-title").textContent = model.title;
  const found = all.find((v) => v.b.id === chainPicked);
  info.innerHTML = "";
  if (!found) {
    const next = all.find((v) => v.b.state !== "built");
    addText(info, "meta", "Clique sur une brique pour savoir à quoi elle sert et où la trouver. Fais glisser la carte pour la faire tourner." +
      (next ? ` Prochaine brique à poser : « ${next.b.label} » (${next.stage}).` : " Toutes les briques sont en place."));
    return;
  }
  const { b, stage, i } = found;
  const head = document.createElement("div");
  head.className = "chain-info-head";
  head.innerHTML = "<b></b><span></span>";
  head.querySelector("b").textContent = `${stage} › ${b.label}`;
  head.querySelector("b").style.color = STAGE_COLOR[i];
  head.querySelector("span").textContent = STATE_TEXT[b.state];
  head.querySelector("span").className = `chain-state ${b.state}`;
  info.appendChild(head);
  [["À quoi ça sert", b.role], ["Dans ton projet", b.value], ["Sans elle", b.without]].forEach(([k, v]) => {
    const p = document.createElement("p");
    p.className = "kv";
    p.innerHTML = "<i></i><span></span>";
    p.querySelector("i").textContent = k;
    p.querySelector("span").textContent = v;
    info.appendChild(p);
  });
  const actions = document.createElement("div");
  actions.className = "train-actions";
  const target = b.target && $(b.target);
  if (target && !target.hidden) {
    const go = document.createElement("button");
    go.className = "small";
    go.textContent = "Voir dans la page ↓";
    go.addEventListener("click", () => target.scrollIntoView({ behavior: "smooth", block: "start" }));
    actions.appendChild(go);
  }
  if (b.file && typeof openFile === "function" && typeof proj !== "undefined" && proj.root === "code") {
    const open = document.createElement("button");
    open.className = "small";
    open.textContent = `Ouvrir ${b.file}`;
    open.addEventListener("click", () => openFile({ name: b.file.split("/").pop(), path: b.file, kind: "file" }));
    actions.appendChild(open);
  }
  if (actions.children.length) info.appendChild(actions);
}

let chainResult = null;

// L'état de l'infrastructure ou de la production vient de changer : on redessine avec la même analyse.
function chainRefresh() { if (chainResult) renderChain(chainResult); }

// Appelé après chaque analyse réussie.
function renderChain(res) {
  chainResult = res;
  chainModel = chainFor(res);
  const ids = chainModel.stages.flatMap((s) => s.blocks.map((b) => b.id));
  if (!ids.includes(chainPicked)) chainPicked = null;
  $("chain-card").hidden = false;
  $("chain-body").hidden = !chainOpen;
  $("chain-toggle").textContent = chainOpen ? "Réduire" : "Déplier";
  if (chainOpen) drawChain(); else drawChainInfo();
}

(() => {
  // Faire tourner la carte en la faisant glisser ; un simple clic garde son rôle (choisir une brique).
  const root = $("chain");
  let from = null;
  root.addEventListener("pointerdown", (e) => { from = { x: e.clientX, yaw: chainYaw, moved: false }; });
  window.addEventListener("pointermove", (e) => {
    if (!from) return;
    if (Math.abs(e.clientX - from.x) < 4 && !from.moved) return;
    from.moved = true;
    chainYaw = Math.max(CHAIN.yawMin, Math.min(CHAIN.yawMax, from.yaw - (e.clientX - from.x) * 0.006));
    drawChain();
  });
  window.addEventListener("pointerup", () => { from = null; });
  root.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    chainYaw = Math.max(CHAIN.yawMin, Math.min(CHAIN.yawMax, chainYaw + (e.key === "ArrowLeft" ? 0.08 : -0.08)));
    drawChain();
  });
  $("chain-toggle").addEventListener("click", () => {
    chainOpen = !chainOpen;
    try { localStorage.setItem("neuro-lens-chaine", chainOpen ? "1" : "0"); } catch (e) { /* stockage indisponible */ }
    $("chain-body").hidden = !chainOpen;
    $("chain-toggle").textContent = chainOpen ? "Réduire" : "Déplier";
    if (chainOpen && chainModel) drawChain();
  });
})();
