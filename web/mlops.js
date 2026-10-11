// neuro-lens : la chaîne MLOps d'un modèle scikit-learn — essais, versions, mise en production, surveillance.
// Tout est réel et local : le déploiement écrit un vrai dossier (modèle, serve.py, Dockerfile),
// l'API répond avec la version en production, la surveillance mesure les requêtes reçues.

let opsFile = "serve.py";   // onglet ouvert dans le dossier de déploiement
let opsAnswer = null;       // dernière réponse de l'API de prédiction
let opsDrift = 0;           // dérive choisie pour la simulation
let opsBusy = "";

const pct = (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)} %`);

async function opsCall(path, body) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return await r.json();
  } catch (e) {
    return { ok: false, error: "Le serveur ne répond pas." };
  }
}

// `ctx` : { code(), override, onHover, onPick } fournis par la page.
function renderMLOps(root, res, ctx) {
  let box = document.getElementById("mlops");
  if (!box || !root.contains(box)) {
    box = document.createElement("div");
    box.id = "mlops";
    root.appendChild(box);
  }
  box.innerHTML = "";
  const ops = res.mlops;
  if (!ops) return;
  const redraw = (next) => { if (next) res.mlops = next; renderMLOps(root, res, ctx); };
  const prod = ops.production, mon = ops.monitoring;
  const stale = prod && prod.run_key !== res.run_key;

  // ----- Le fil de la chaîne -----
  const chain = document.createElement("div");
  chain.className = "ops-chain";
  const stages = [
    ["Données", `empreinte ${res.data_id}`, "ok", null],
    ["Entraînement", res.estimator, "ok", null],
    ["Essais", `${ops.runs.length} noté${ops.runs.length > 1 ? "s" : ""}`, "ok", "ops-runs"],
    ["Version", ops.versions.length ? `${ops.versions.length} au registre` : "aucune", ops.versions.length ? "ok" : "todo", "ops-deploy"],
    ["Production", prod ? `v${prod.version} en service` : "rien en service", prod ? "ok" : "todo", "ops-api"],
    ["Surveillance", !mon ? "pas de trafic" : mon.status === "alert" ? "dérive !" : mon.status === "watch" ? "à surveiller" : "tout va bien",
      !mon ? "todo" : mon.status === "ok" ? "ok" : mon.status, "ops-watch"],
  ];
  stages.forEach(([name, sub, status, target], i) => {
    if (i) chain.appendChild(Object.assign(document.createElement("i"), { textContent: "→" }));
    const s = document.createElement("button");
    s.className = `ops-stage ${status}`;
    s.innerHTML = "<b></b><span></span>";
    s.querySelector("b").textContent = `${i + 1}. ${name}`;
    s.querySelector("span").textContent = sub;
    if (target) s.addEventListener("click", () => document.getElementById(target)?.scrollIntoView({ behavior: "smooth", block: "start" }));
    chain.appendChild(s);
  });
  const intro = mlCard("attn", "De l'essai à la production", "chaîne MLOps");
  addText(intro, "explain", "Entraîner un modèle n'est que le début. Pour qu'il serve à quelqu'un, il faut garder la trace des essais, figer une version, la mettre derrière une API, puis vérifier qu'elle tient dans le temps. Chaque étape ci-dessous est réelle et tourne sur ta machine.");
  intro.appendChild(chain);
  box.appendChild(intro);

  // ----- 3. Suivi des essais -----
  const runs = mlCard("dense", "Suivi des essais", `${ops.runs.length} essai${ops.runs.length > 1 ? "s" : ""}`);
  runs.id = "ops-runs";
  addText(runs, "explain", "Chaque combinaison modèle + réglage + données que tu essaies est notée ici, avec ses scores. Sans cette trace, impossible de dire dans un mois pourquoi cette version a été choisie. C'est le rôle d'outils comme MLflow ou Weights & Biases. Change le réglage ou le modèle : une ligne s'ajoute.");
  const table = document.createElement("table");
  table.className = "ops-table";
  table.innerHTML = "<tr><th>n°</th><th>heure</th><th>modèle</th><th>réglage</th><th>données</th><th>score appris</th><th>score mis de côté</th></tr>";
  ops.runs.forEach((r) => {
    const tr = table.insertRow();
    if (r.key === res.run_key) tr.className = "current";
    [r.id, r.at, r.estimator, r.setting, r.data_id, pct(r.train), pct(r.held_out) + (r.best ? " ★" : "")].forEach((v) => { tr.insertCell().textContent = v; });
  });
  runs.appendChild(table);
  addText(runs, "meta", "Ligne surlignée : le modèle affiché en ce moment. ★ : meilleur score sur les exemples mis de côté, le seul qui compte pour choisir.");
  box.appendChild(runs);

  // ----- 4. Registre et déploiement -----
  const dep = mlCard("pool", "Versions et mise en production", prod ? `v${prod.version} en production` : "rien en production");
  dep.id = "ops-deploy";
  addText(dep, "explain", "Déployer fige le modèle entraîné dans une version numérotée et écrit un dossier complet : le modèle, sa fiche (d'où il vient, ses scores), le code du service et la recette de l'image Docker. Une version ne change plus jamais : on peut comparer, et revenir en arrière.");
  const actions = document.createElement("div");
  actions.className = "train-actions";
  const go = document.createElement("button");
  go.textContent = opsBusy === "deploy" ? "Déploiement…" : `Déployer ce modèle (version ${(ops.versions.at(-1)?.version ?? 0) + 1})`;
  go.disabled = Boolean(opsBusy);
  go.addEventListener("click", async () => {
    opsBusy = "deploy"; redraw();
    const out = await opsCall("/api/deploy", { code: ctx.code(), override: ctx.override() });
    opsBusy = ""; opsAnswer = null;
    if (!out.ok) { redraw(); return alertIn(dep, out.error); }
    redraw(out.mlops);
  });
  actions.appendChild(go);
  dep.appendChild(actions);
  if (stale) addText(dep, "meta", `Le modèle affiché n'est pas celui en production (v${prod.version} : ${prod.estimator}). Déploie pour le remplacer.`).classList.add("ops-warn");
  if (ops.versions.length) {
    const vt = document.createElement("table");
    vt.className = "ops-table";
    vt.innerHTML = "<tr><th>version</th><th>créée</th><th>modèle</th><th>score appris</th><th>score mis de côté</th><th>taille</th><th></th></tr>";
    ops.versions.slice().reverse().forEach((v) => {
      const tr = vt.insertRow();
      const live = prod && v.version === prod.version;
      if (live) tr.className = "current";
      [`v${v.version}`, v.created, v.estimator, pct(v.score_train), pct(v.score_held_out), `${v.size_kb} ko`, live ? "en production" : ""].forEach((x) => { tr.insertCell().textContent = x; });
    });
    dep.appendChild(vt);
  }
  if (prod) {
    addText(dep, "meta", `Dossier de la version en production : ${prod.folder}`).classList.add("mono");
    const tabs = document.createElement("div");
    tabs.className = "ops-tabs";
    const names = Object.keys(prod.texts);
    if (!names.includes(opsFile)) opsFile = names[0];
    names.forEach((name) => {
      const t = document.createElement("button");
      t.className = name === opsFile ? "small on" : "small";
      t.textContent = name;
      t.addEventListener("click", () => { opsFile = name; redraw(); });
      tabs.appendChild(t);
    });
    dep.appendChild(tabs);
    const pre = document.createElement("pre");
    pre.className = "ops-code";
    pre.textContent = prod.texts[opsFile];
    dep.appendChild(pre);
    addText(dep, "meta", {
      "serve.py": "Le service : il charge le modèle une fois au démarrage, puis répond aux demandes. /health sert à l'orchestrateur pour savoir s'il est vivant.",
      "Dockerfile": "La recette de l'image : même Python, mêmes bibliothèques, même modèle partout où elle tourne. Je ne l'ai pas construite ici ; la commande est en tête du fichier.",
      "requirements.txt": "Les versions exactes des bibliothèques : un modèle enregistré avec une version de scikit-learn peut ne pas se recharger avec une autre.",
      "model_card.json": "La fiche du modèle : qui il est, sur quelles données il a appris (empreinte), avec quels réglages et quels scores.",
    }[opsFile] ?? "");
  }
  box.appendChild(dep);
  if (!prod) return;

  // ----- 5. L'API de prédiction -----
  const api = mlCard("conv", "API de prédiction", `POST /api/predict · v${prod.version}`);
  api.id = "ops-api";
  addText(api, "explain", "La version en production répond ici, comme elle le ferait pour une application. Change les valeurs, envoie, et regarde la réponse et le temps mis à répondre.");
  const form = document.createElement("div");
  form.className = "ops-form";
  const inputs = prod.features.map((name, i) => {
    const label = document.createElement("label");
    label.textContent = name;
    const input = document.createElement("input");
    input.type = "number"; input.step = "any";
    input.value = opsAnswer?.sent?.[i] ?? prod.example[i];
    label.appendChild(input);
    form.appendChild(label);
    return input;
  });
  const send = document.createElement("button");
  send.textContent = "Envoyer";
  form.appendChild(send);
  api.appendChild(form);
  const curl = document.createElement("pre");
  curl.className = "ops-code";
  const showCurl = () => {
    curl.textContent = `curl -X POST ${location.origin}/api/predict \\\n  -H "Content-Type: application/json" \\\n  -d '{"rows": [[${inputs.map((x) => Number(x.value) || 0).join(", ")}]]}'`;
  };
  showCurl();
  inputs.forEach((x) => x.addEventListener("input", showCurl));
  api.appendChild(curl);
  send.addEventListener("click", async () => {
    const row = inputs.map((x) => Number(x.value) || 0);
    const out = await opsCall("/api/predict", { rows: [row] });
    opsAnswer = { ...out, sent: row };
    const again = await opsCall("/api/simulate", { n: 0, drift: 0 }); // l'état à jour, sans trafic en plus
    redraw(again.ok ? again.mlops : null);
  });
  if (opsAnswer) {
    if (opsAnswer.ok) {
      const cls = opsAnswer.predictions[0];
      const p = opsAnswer.probabilities?.[0];
      addText(api, "ops-answer", `Réponse : ${cls}` + (p ? ` (sûr à ${Math.round(Math.max(...p) * 100)} %)` : "") +
        ` · version ${opsAnswer.version} · ${opsAnswer.latency_ms} ms`);
    } else addText(api, "meta", opsAnswer.error).classList.add("ops-warn");
  }
  box.appendChild(api);

  // ----- 6. Surveillance -----
  const watch = mlCard(mon && mon.status !== "ok" ? "act" : "pool", "Surveillance en production",
    mon ? `${mon.requests} requête${mon.requests > 1 ? "s" : ""}` : "pas encore de trafic");
  watch.id = "ops-watch";
  addText(watch, "explain", "Un modèle en production ne plante pas quand il devient mauvais : il continue de répondre, mais faux. La cause la plus fréquente est la dérive : les données reçues ne ressemblent plus à celles de l'entraînement. Simule du trafic, puis augmente la dérive et recommence.");
  const sim = document.createElement("div");
  sim.className = "train-actions";
  const slider = document.createElement("input");
  slider.type = "range"; slider.min = 0; slider.max = 3; slider.step = 0.25; slider.value = opsDrift;
  slider.className = "knob ops-drift";
  slider.setAttribute("aria-label", "Dérive des données");
  const said = document.createElement("span");
  said.className = "meta";
  const say = () => { said.textContent = Number(slider.value) ? `dérive : ${slider.value} écart-type` : "dérive : aucune (le monde n'a pas changé)"; };
  say();
  slider.addEventListener("input", () => { opsDrift = Number(slider.value); say(); });
  const fire = document.createElement("button");
  fire.textContent = opsBusy === "simulate" ? "Envoi…" : "Simuler 200 requêtes";
  fire.disabled = Boolean(opsBusy);
  fire.addEventListener("click", async () => {
    opsBusy = "simulate"; redraw();
    const out = await opsCall("/api/simulate", { n: 200, drift: opsDrift });
    opsBusy = "";
    redraw(out.ok ? out.mlops : null);
  });
  sim.append(fire, slider, said);
  watch.appendChild(sim);
  if (mon) {
    const banner = addText(watch, `ops-banner ${mon.status}`, {
      ok: "Tout va bien : les données reçues ressemblent à celles de l'entraînement.",
      watch: "À surveiller : les données reçues commencent à s'écarter de celles de l'entraînement.",
      alert: "Alerte dérive : les données reçues ne ressemblent plus à celles de l'entraînement. Les réponses du modèle ne sont plus fiables ; il faut le réentraîner sur des données récentes.",
    }[mon.status]);
    banner.setAttribute("role", "status");
    const kpis = document.createElement("div");
    kpis.className = "ops-kpis";
    const kpi = (label, value, note) => {
      const k = document.createElement("div");
      k.innerHTML = "<span></span><b></b><i></i>";
      k.querySelector("span").textContent = label;
      k.querySelector("b").textContent = value;
      k.querySelector("i").textContent = note ?? "";
      kpis.appendChild(k);
    };
    kpi("Requêtes reçues", mon.requests);
    kpi("Latence", `${mon.latency_p50} ms`, `95 % sous ${mon.latency_p95} ms`);
    if (mon.quality !== undefined) kpi("Bonnes réponses", pct(mon.quality), `${pct(mon.quality_at_deploy)} au déploiement`);
    kpi("Dérive (PSI)", mon.drift.toFixed(2), "alerte à 0,25");
    watch.appendChild(kpis);
    if (mon.quality_curve?.length > 1) watch.appendChild(qualityCurve(mon));
    const grid = document.createElement("div");
    grid.className = "ops-hists";
    mon.features.forEach((f) => grid.appendChild(driftHistogram(f)));
    watch.appendChild(grid);
    watch.appendChild(legendOf([["données d'entraînement", "#8b93a7"], ["données reçues en production", classColor(3)]]));
    addText(watch, "meta", "En vrai, on ne connaît presque jamais la bonne réponse tout de suite : la qualité ne se mesure qu'après coup. La dérive, elle, se voit immédiatement, d'où son intérêt comme signal d'alarme. Ici les bonnes réponses sont connues parce que le trafic est simulé à partir des données d'entraînement.");
  }
  box.appendChild(watch);
}

function alertIn(card, text) {
  addText(card, "meta", text).classList.add("ops-warn");
}

// Bonnes réponses par tranche de 50 requêtes : on voit le moment où ça décroche.
function qualityCurve(mon) {
  const W = 420, H = 110, padL = 34, padB = 16, padT = 8;
  const [cv, ctx] = plotArea(W, H);
  const pts = mon.quality_curve;
  const x = (i) => padL + (i / Math.max(1, pts.length - 1)) * (W - padL - 10);
  const y = (v) => H - padB - v * (H - padB - padT);
  ctx.font = "10px system-ui"; ctx.fillStyle = "#8b93a7"; ctx.strokeStyle = "#2a3142"; ctx.textAlign = "right";
  [0, 0.5, 1].forEach((v) => { ctx.beginPath(); ctx.moveTo(padL, y(v)); ctx.lineTo(W - 10, y(v)); ctx.stroke(); ctx.fillText(`${v * 100} %`, padL - 4, y(v) + 3); });
  if (mon.quality_at_deploy) {
    ctx.strokeStyle = "#8b93a7"; ctx.setLineDash([4, 4]); ctx.beginPath();
    ctx.moveTo(padL, y(mon.quality_at_deploy)); ctx.lineTo(W - 10, y(mon.quality_at_deploy)); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = classColor(2); ctx.fillStyle = classColor(2); ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.stroke();
  pts.forEach((v, i) => { ctx.beginPath(); ctx.arc(x(i), y(v), 2.5, 0, 2 * Math.PI); ctx.fill(); });
  const fig = document.createElement("figure");
  fig.className = "ops-figure";
  fig.appendChild(cv);
  const cap = document.createElement("figcaption");
  cap.textContent = "Bonnes réponses par tranche de 50 requêtes, de la plus ancienne à la plus récente. Pointillés : le niveau au déploiement.";
  fig.appendChild(cap);
  return fig;
}

// Une colonne du modèle : la forme des données d'entraînement (gris) contre celle des données reçues (jaune).
function driftHistogram(f) {
  const W = 200, H = 84, pad = 4;
  const [cv, ctx] = plotArea(W, H);
  const max = Math.max(...f.reference, ...f.received) || 1;
  const bw = (W - 2 * pad) / f.reference.length;
  const bar = (values, color, inset) => {
    ctx.fillStyle = color;
    values.forEach((v, i) => { const h = (v / max) * (H - 2 * pad); ctx.fillRect(pad + i * bw + inset, H - pad - h, bw - 1 - 2 * inset, h); });
  };
  bar(f.reference, "#8b93a799", 0);
  bar(f.received, classColor(3) + "cc", 2);
  const fig = document.createElement("figure");
  fig.className = "ops-figure";
  fig.appendChild(cv);
  const cap = document.createElement("figcaption");
  const level = f.psi >= 0.25 ? "alert" : f.psi >= 0.1 ? "watch" : "ok";
  cap.innerHTML = `<b></b> de ${f.range[0]} à ${f.range[1]} · <span class="ops-psi ${level}">PSI ${f.psi.toFixed(2)}</span>`;
  cap.querySelector("b").textContent = f.name;
  fig.appendChild(cap);
  return fig;
}
