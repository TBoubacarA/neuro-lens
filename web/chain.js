// neuro-lens : la carte 3D de la chaîne, de la donnée à la production.
//
// Quatre plaques — Données, Modèle, Infrastructure, Production — alignées comme les couches d'un
// réseau, et sur chacune ce que l'étage contient vraiment dans le projet. Sous la carte, les briques
// de chaque étage avec leur état (en place, prête, absente) ; un clic en explique une simplement :
// à quoi elle sert, ce qu'elle vaut dans ce projet, ce qui se passe sans elle, où la trouver.

const STATE_TEXT = { built: "en place", ready: "prête, pas encore branchée", missing: "pas encore là" };
const STAGE_COLOR = ["#ff9a85", "#b083f0", "#5b8cff", "#4cd98a"];

let chainYaw = 0.78;
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

// ---------- Le dessin 3D : une plaque par étage, avec le vrai contenu du projet ----------
//
// Chaque étage est une plaque dressée, et sur la plaque on dessine ce qu'il contient vraiment :
// les points des données, la frontière apprise, les versions et le conteneur, le trafic reçu.
// Les plaques s'alignent dans l'ordre de la chaîne, comme les couches d'un réseau en 3D.

const PLATE = { w: 3.6, h: 2.5, gap: 1.9, tex: [320, 222] };
let chainPitch = 0.32;
let chainHits = [];        // quadrilatères projetés, pour savoir sur quelle plaque on clique
let chainStage = null;     // étage choisi (ses briques sont mises en avant sous la carte)

function plateProject([x, y, z], cw, ch, scale) { // projection parallèle : une plaque reste un parallélogramme
  const c = Math.cos(chainYaw), s = Math.sin(chainYaw);
  const x1 = x * c + z * s, z1 = -x * s + z * c;
  const y2 = y * Math.cos(chainPitch) - z1 * Math.sin(chainPitch);
  const z2 = y * Math.sin(chainPitch) + z1 * Math.cos(chainPitch);
  return [cw / 2 + x1 * scale, ch / 2 + y2 * scale, z2];
}

// Le contenu de chaque plaque, dessiné sur une petite toile hors écran.
function plateTexture(res, model, i) {
  const [W, H] = PLATE.tex, cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d"), color = STAGE_COLOR[i], stage = model.stages[i];
  const built = stage.blocks.filter((b) => b.state === "built").length;
  ctx.fillStyle = "#151a24"; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = color; ctx.fillRect(0, 0, W, 30);
  ctx.fillStyle = "#0f1117"; ctx.font = "bold 15px system-ui"; ctx.textBaseline = "middle";
  ctx.fillText(`${i + 1}. ${stage.name}`, 10, 15);
  ctx.font = "12px system-ui"; ctx.textAlign = "right";
  ctx.fillText(`${built} / ${stage.blocks.length} en place`, W - 10, 15);
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  const box = { x: 12, y: 40, w: W - 24, h: H - 52 };
  const drawn = (res.kind === "ml" ? texML : res.kind === "agent" ? texAgent : texTorch)(ctx, res, i, box);
  if (!drawn) { // rien à montrer : l'étage n'existe pas encore dans ce projet
    ctx.strokeStyle = color; ctx.setLineDash([6, 5]); ctx.strokeRect(box.x, box.y, box.w, box.h); ctx.setLineDash([]);
    ctx.fillStyle = "#8b93a7"; ctx.font = "14px system-ui"; ctx.textAlign = "center";
    ctx.fillText("pas encore là", W / 2, box.y + box.h / 2);
    ctx.textAlign = "left";
  }
  return cv;
}

function texText(ctx, lines, box, size = 13) {
  ctx.font = `${size}px ui-monospace, Menlo, monospace`;
  lines.slice(0, Math.floor(box.h / (size + 6))).forEach(([text, c], k) => {
    ctx.fillStyle = c || "#e6e8ee";
    ctx.fillText(String(text).slice(0, 34), box.x + 4, box.y + 14 + k * (size + 6));
  });
  return true;
}

function texPoints(ctx, sc, box, withRegions) {
  const xs = sc.points.map((p) => p[0]), ys = sc.points.map((p) => p[1]);
  const [x0, x1] = sc.regions ? sc.regions.x : [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = sc.regions ? sc.regions.y : [Math.min(...ys), Math.max(...ys)];
  const X = (v) => box.x + ((v - x0) / (x1 - x0 || 1)) * box.w, Y = (v) => box.y + box.h - ((v - y0) / (y1 - y0 || 1)) * box.h;
  if (withRegions && sc.regions) {
    const n = sc.regions.cells.length, cw = box.w / n, chh = box.h / n;
    sc.regions.cells.forEach((row, r) => row.forEach((c, k) => {
      ctx.fillStyle = classColor(c) + "55";
      ctx.fillRect(box.x + k * cw, box.y + box.h - (r + 1) * chh, cw + 0.6, chh + 0.6);
    }));
  }
  sc.points.forEach(([x, y], k) => {
    ctx.fillStyle = classColor(sc.labels[k]);
    ctx.beginPath(); ctx.arc(X(x), Y(y), 2.6, 0, 2 * Math.PI); ctx.fill();
  });
  return true;
}

function texBars(ctx, values, box, colors) {
  const max = Math.max(...values.map(Math.abs), 1e-9), bw = box.w / values.length;
  values.forEach((v, k) => {
    const h = (Math.abs(v) / max) * box.h;
    ctx.fillStyle = colors ? colors(k, v) : "#5b8cff";
    ctx.fillRect(box.x + k * bw + 1, box.y + box.h - h, Math.max(1, bw - 2), h);
  });
  return true;
}

function texML(ctx, res, i, box) {
  const ops = res.mlops || {}, prod = ops.production, mon = ops.monitoring;
  if (i === 0) return res.scatter ? texPoints(ctx, res.scatter, box, false)
    : res.curve ? texPoints(ctx, { points: res.curve.points, labels: res.curve.points.map(() => 0) }, box, false)
    : texText(ctx, [[`${res.n_samples} exemples`], [`${res.features.length} colonnes`]], box);
  if (i === 1) {
    if (res.scatter && res.scatter.regions) return texPoints(ctx, res.scatter, box, true);
    const bars = res.importances || res.coefficients;
    return bars ? texBars(ctx, bars.map((b) => b.value), box, (k, v) => (v >= 0 ? "#b083f0" : "#ff6b6b")) : texText(ctx, [[res.estimator]], box);
  }
  if (i === 2) {
    const infra = typeof opsInfra !== "undefined" && opsInfra ? opsInfra : { phase: "idle" };
    const lines = [["tests/test_train.py", "#4cd98a"], [".github/workflows/tests.yml", "#f2b84b"]];
    (ops.versions || []).slice(-3).forEach((v) => lines.push([`v${v.version}  ${v.estimator}`, prod && prod.version === v.version ? "#4cd98a" : "#e6e8ee"]));
    if (["starting", "running", "stopped"].includes(infra.phase)) lines.push([`image ${infra.size_mb} Mo`, "#5b8cff"]);
    return texText(ctx, lines, box, 12);
  }
  if (!prod) return false;
  if (mon && mon.features && mon.features.length) { // le trafic reçu (jaune) contre les données d'entraînement (gris)
    const f = mon.features[0], half = { ...box, h: box.h - 18 };
    const n = f.reference.length, bw = half.w / n, max = Math.max(...f.reference, ...f.received, 1e-9);
    f.reference.forEach((v, k) => { ctx.fillStyle = "#8b93a799"; const h = (v / max) * half.h; ctx.fillRect(half.x + k * bw, half.y + half.h - h, bw - 2, h); });
    f.received.forEach((v, k) => { ctx.fillStyle = "#f2b84bcc"; const h = (v / max) * half.h; ctx.fillRect(half.x + k * bw + 3, half.y + half.h - h, bw - 6, h); });
    ctx.fillStyle = mon.status === "alert" ? "#ff6b6b" : "#8b93a7"; ctx.font = "12px system-ui";
    ctx.fillText(`${f.name} · ${mon.requests} requêtes · dérive ${mon.drift.toFixed(2)}`, box.x, box.y + box.h);
    return true;
  }
  return texText(ctx, [[`API v${prod.version} en service`, "#4cd98a"], ["aucun trafic observé", "#8b93a7"]], box);
}

function texTorch(ctx, res, i, box) {
  if (i === 0) {
    const map = res.input.preview && res.input.preview.maps && res.input.preview.maps[0];
    if (!map) return texText(ctx, (res.input.tokens || []).map((t) => [t]), box, 14);
    const n = map.length, cell = Math.min(box.w, box.h) / n, ox = box.x + (box.w - cell * n) / 2;
    map.forEach((row, r) => row.forEach((v, k) => { const g = Math.round(v * 255); ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(ox + k * cell, box.y + r * cell, cell + 0.5, cell + 0.5); }));
    return true;
  }
  if (i === 1) {
    const conv = res.layers.find((l) => l.preview && l.preview.maps);
    if (conv) { // les cartes d'activation de la première couche qui voit l'image
      const maps = conv.preview.maps.slice(0, 8), cols = 4, size = Math.min(box.w / cols, (box.h - 30) / 2) - 4;
      maps.forEach((m, k) => {
        const n = m.length, cell = size / n, ox = box.x + (k % cols) * (size + 4), oy = box.y + Math.floor(k / cols) * (size + 4);
        m.forEach((row, r) => row.forEach((v, q) => { ctx.fillStyle = `rgb(${heat(v).slice(0, 3).join(",")})`; ctx.fillRect(ox + q * cell, oy + r * cell, cell + 0.4, cell + 0.4); }));
      });
    }
    const probs = res.prediction && res.prediction.probs;
    if (probs) texBars(ctx, probs, { x: box.x, y: box.y + box.h - 26, w: box.w, h: 26 }, (k) => (k === res.prediction.top ? "#4cd98a" : "#b083f0"));
    if (!conv) { // un réseau de texte : les vecteurs de la première couche, un mot par ligne
      const rows = res.layers[0] && res.layers[0].preview.rows;
      if (!rows) return texText(ctx, [[`${res.layers.length} couches`]], box);
      const area = probs ? { ...box, h: box.h - 32 } : box, cw = area.w / rows[0].length, chh = area.h / rows.length;
      rows.forEach((row, r) => row.forEach((v, k) => { const a = Math.min(1, Math.abs(v)); ctx.fillStyle = v >= 0 ? `rgba(242,160,75,${a})` : `rgba(91,140,255,${a})`; ctx.fillRect(area.x + k * cw, area.y + r * chh, cw - 1, chh - 1); }));
    }
    return true;
  }
  if (i === 2) return texText(ctx, [["tests/test_model.py", "#4cd98a"], [".github/workflows/tests.yml", "#f2b84b"], ["pas encore de version figée", "#8b93a7"]], box, 12);
  return false;
}

function texAgent(ctx, res, i, box) {
  const ops = res.ops || {};
  if (i === 0) return res.start ? texText(ctx, Object.entries(res.start).map(([k, v]) => [`${k}: ${v}`]), box, 12) : false;
  if (i === 1) { // les étapes du graphe, avec leurs chemins
    const steps = res.nodes.filter((n) => n.kind === "step"), pos = {};
    steps.forEach((n, k) => { pos[n.id] = [box.x + 20 + k * ((box.w - 40) / Math.max(1, steps.length - 1)), box.y + box.h / 2 + (k % 2 ? 26 : -26)]; });
    ctx.strokeStyle = "#8b93a7"; ctx.lineWidth = 1.2;
    res.edges.forEach((e) => { const a = pos[e.source], b = pos[e.target]; if (a && b) { ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); } });
    steps.forEach((n) => {
      const [x, y] = pos[n.id];
      ctx.fillStyle = "#f2b84b"; ctx.beginPath(); ctx.arc(x, y, 9, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#e6e8ee"; ctx.font = "11px system-ui"; ctx.textAlign = "center"; ctx.fillText(n.id.slice(0, 10), x, y + 24); ctx.textAlign = "left";
    });
    return true;
  }
  if (i === 2) {
    const lines = [];
    (ops.rules || []).forEach((r) => lines.push([`garde-fou : ${r}`, "#4cd98a"]));
    if (ops.limits) Object.entries(ops.limits).forEach(([k, v]) => lines.push([`plafond ${k} : ${v}`, "#f2b84b"]));
    if (res.start) lines.push(["tests/test_agent.py", "#4cd98a"]);
    return lines.length ? texText(ctx, lines, box, 12) : false;
  }
  const spans = ops.spans || [];
  if (!spans.length) return false;
  const t0 = Math.min(...spans.map((s) => s.start)), t1 = Math.max(...spans.map((s) => s.end)), rowH = Math.min(18, box.h / spans.length);
  spans.forEach((sp, k) => { // la trace : une barre par étape, à sa place dans le temps
    const x = box.x + ((sp.start - t0) / (t1 - t0 || 1)) * box.w, w = Math.max(3, ((sp.end - sp.start) / (t1 - t0 || 1)) * box.w);
    ctx.fillStyle = sp.kind === "step" ? "#4cd98a" : "#5b8cff"; ctx.fillRect(x, box.y + k * rowH, w, rowH - 3);
  });
  return true;
}

function drawChain() {
  const root = $("chain"), model = chainModel, res = chainResult;
  if (!model || !res) return;
  let cv = root.querySelector("canvas");
  if (!cv) { root.innerHTML = ""; cv = document.createElement("canvas"); cv.className = "chain-canvas"; root.appendChild(cv); }
  const cw = Math.max(320, root.clientWidth || 600), ch = 360, dpr = window.devicePixelRatio || 1;
  cv.width = cw * dpr; cv.height = ch * dpr; cv.style.width = `${cw}px`; cv.style.height = `${ch}px`;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const n = model.stages.length, scale = Math.min(cw / 8.2, 80);
  const zOf = (i) => (i - (n - 1) / 2) * PLATE.gap;
  const P = (p) => plateProject(p, cw, ch, scale);
  const textures = model.stages.map((_, i) => plateTexture(res, model, i));

  // Le fil de la chaîne : il traverse les plaques dans l'ordre, des données jusqu'à la production.
  ctx.strokeStyle = "#8b93a7"; ctx.setLineDash([5, 5]); ctx.lineWidth = 1.2;
  ctx.beginPath();
  model.stages.forEach((_, i) => { const [x, y] = P([0, 0, zOf(i)]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
  ctx.stroke(); ctx.setLineDash([]);

  // Les plaques, de la plus lointaine à la plus proche.
  const order = model.stages.map((_, i) => ({ i, depth: P([0, 0, zOf(i)])[2] })).sort((a, b) => b.depth - a.depth);
  chainHits = [];
  order.forEach(({ i }) => {
    const z = zOf(i), w = PLATE.w / 2, h = PLATE.h / 2;
    const tl = P([-w, -h, z]), tr = P([w, -h, z]), bl = P([-w, h, z]), br = P([w, h, z]);
    const [tw, th] = PLATE.tex;
    ctx.save();
    ctx.globalAlpha = chainStage === null || chainStage === i ? 0.95 : 0.45;
    ctx.setTransform(dpr * (tr[0] - tl[0]) / tw, dpr * (tr[1] - tl[1]) / tw, dpr * (bl[0] - tl[0]) / th, dpr * (bl[1] - tl[1]) / th, dpr * tl[0], dpr * tl[1]);
    ctx.drawImage(textures[i], 0, 0);
    ctx.restore();
    ctx.strokeStyle = chainStage === i ? "#ffffff" : STAGE_COLOR[i];
    ctx.lineWidth = chainStage === i ? 2.2 : 1.2;
    ctx.beginPath();
    [tl, tr, br, bl].forEach(([x, y], k) => { if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.closePath(); ctx.stroke();
    chainHits.push({ i, quad: [tl, tr, br, bl] });
  });
  drawChainChips();
  drawChainInfo();
}

// Sous la carte : les briques de chaque étage, avec leur état. Un clic l'explique.
function drawChainChips() {
  const box = $("chain-chips"), model = chainModel;
  box.innerHTML = "";
  model.stages.forEach((stage, i) => {
    const row = document.createElement("div");
    row.className = `chain-row${chainStage === i ? " on" : ""}`;
    const name = document.createElement("b");
    name.textContent = `${i + 1}. ${stage.name}`;
    name.style.color = STAGE_COLOR[i];
    row.appendChild(name);
    stage.blocks.forEach((b) => {
      const chip = document.createElement("button");
      chip.className = `chain-chip ${b.state}${b.id === chainPicked ? " picked" : ""}`;
      chip.textContent = `${b.state === "built" ? "■" : b.state === "ready" ? "▣" : "□"} ${b.label}`;
      chip.title = `${STATE_TEXT[b.state]} · ${b.value}`;
      chip.addEventListener("click", () => { chainPicked = b.id; chainStage = i; drawChain(); });
      row.appendChild(chip);
    });
    box.appendChild(row);
  });
}

function insideQuad([x, y], quad) {
  let inside = false;
  for (let a = 0, b = quad.length - 1; a < quad.length; b = a++) {
    const [xa, ya] = quad[a], [xb, yb] = quad[b];
    if ((ya > y) !== (yb > y) && x < ((xb - xa) * (y - ya)) / (yb - ya) + xa) inside = !inside;
  }
  return inside;
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
  const before = chainModel && chainModel.title;
  chainResult = res;
  chainModel = chainFor(res);
  if (chainModel.title !== before) chainStage = null; // autre type de projet : aucune plaque mise en avant
  const ids = chainModel.stages.flatMap((s) => s.blocks.map((b) => b.id));
  if (!ids.includes(chainPicked)) chainPicked = null;
  $("chain-card").hidden = false;
  $("chain-body").hidden = !chainOpen;
  $("chain-toggle").textContent = chainOpen ? "Réduire" : "Déplier";
  if (chainOpen) drawChain(); else drawChainInfo();
}

(() => {
  // Faire glisser fait tourner la carte ; un simple clic choisit la plaque sous la souris.
  const root = $("chain");
  let from = null;
  root.addEventListener("pointerdown", (e) => { from = { x: e.clientX, y: e.clientY, yaw: chainYaw, pitch: chainPitch, moved: false }; });
  window.addEventListener("pointermove", (e) => {
    if (!from) return;
    if (Math.hypot(e.clientX - from.x, e.clientY - from.y) < 4 && !from.moved) return;
    from.moved = true;
    chainYaw = Math.max(0.05, Math.min(1.45, from.yaw - (e.clientX - from.x) * 0.006));
    chainPitch = Math.max(0.05, Math.min(0.75, from.pitch + (e.clientY - from.y) * 0.004));
    drawChain();
  });
  window.addEventListener("pointerup", (e) => {
    if (from && !from.moved) {
      const canvas = root.querySelector("canvas");
      if (canvas) {
        const r = canvas.getBoundingClientRect(), pt = [e.clientX - r.left, e.clientY - r.top];
        const hit = chainHits.slice().reverse().find((h) => insideQuad(pt, h.quad)); // la plus proche d'abord
        chainStage = hit ? hit.i : null;
        if (hit) chainPicked = null;
        drawChain();
      }
    }
    from = null;
  });
  root.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    chainYaw = Math.max(0.05, Math.min(1.45, chainYaw + (e.key === "ArrowLeft" ? 0.08 : -0.08)));
    drawChain();
  });
  window.addEventListener("resize", () => { if (chainOpen && chainModel) drawChain(); });
  $("chain-toggle").addEventListener("click", () => {
    chainOpen = !chainOpen;
    try { localStorage.setItem("neuro-lens-chaine", chainOpen ? "1" : "0"); } catch (e) { /* stockage indisponible */ }
    $("chain-body").hidden = !chainOpen;
    $("chain-toggle").textContent = chainOpen ? "Réduire" : "Déplier";
    if (chainOpen && chainModel) drawChain();
  });
})();
