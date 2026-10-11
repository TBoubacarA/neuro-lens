// Carte « Entraînement » : fait apprendre le réseau époque par époque et trace la courbe d'erreur.
// Après chaque époque on relance l'analyse : toutes les vues montrent alors le réseau entraîné.

let training = false;

function renderTraining(res) {
  const card = document.getElementById("train-card");
  card.hidden = !res.trainable;
  document.getElementById("dream-card").hidden = !res.trainable;
  if (!res.trainable) return;
  drawTraining(res.training);
}

function drawTraining(t) {
  const shape = document.getElementById("train-shape");
  const status = document.getElementById("train-status");
  const chart = document.getElementById("train-chart");
  if (!t || !t.epochs.length) {
    shape.textContent = "poids aléatoires";
    if (!training) status.textContent = "";
    chart.innerHTML = "";
    document.getElementById("epoch-row").hidden = true;
    document.getElementById("train-eval").innerHTML = "";
    return;
  }
  const last = t.epochs[t.epochs.length - 1];
  const shown = t.shown ?? t.epochs.length;
  shape.textContent = shown === t.epochs.length
    ? `${t.epochs.length} époque${t.epochs.length > 1 ? "s" : ""}`
    : `revu : après ${shown} époque${shown > 1 ? "s" : ""}`;
  if (!training) {
    status.textContent = `${Math.round(last.test_acc * 100)} % de bonnes réponses sur des chiffres jamais vus`;
  }
  drawEpochSlider(t, shown);
  drawEval(t.eval);

  // Courbe de l'erreur (loss) lot après lot, avec une graduation par époque.
  const W = 520, H = 130, P = 28;
  const L = t.losses;
  const max = Math.max(...L, 0.1);
  const x = (i) => P + (i / Math.max(1, L.length - 1)) * (W - 2 * P);
  const y = (v) => H - P + 4 - (v / max) * (H - 2 * P);
  const pts = L.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  let seps = "";
  let acc = "";
  const perEpoch = L.length / t.epochs.length;
  t.epochs.forEach((e, k) => {
    const xe = x(Math.min(L.length - 1, Math.round((k + 1) * perEpoch) - 1));
    seps += `<line x1="${xe}" x2="${xe}" y1="${P - 10}" y2="${H - P + 4}" class="sep"/>`;
    acc += `<text x="${xe - 3}" y="${P - 14}" text-anchor="end" class="acc">${Math.round(e.test_acc * 100)} %</text>`;
  });
  chart.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" class="loss-chart" role="img" aria-label="Courbe d'erreur pendant l'entraînement">
      <line x1="${P}" x2="${W - P}" y1="${H - P + 4}" y2="${H - P + 4}" class="axis"/>
      ${seps}${acc}
      <polyline points="${pts}" class="loss"/>
      <text x="${P}" y="${H - 6}" class="lbl">erreur (loss) à chaque lot de 64 chiffres · % = réussite sur des chiffres jamais vus</text>
    </svg>`;
}

// Curseur pour revoir le réseau tel qu'il était après chaque époque (0 = avant d'apprendre).
function drawEpochSlider(t, shown) {
  const row = document.getElementById("epoch-row");
  const range = document.getElementById("epoch-range");
  row.hidden = t.kept < 2;
  range.max = t.kept - 1;
  range.value = Math.min(shown, t.kept - 1);
  const acc = shown === 0 ? "poids de départ" : `${Math.round(t.epochs[shown - 1].test_acc * 100)} % de réussite`;
  document.getElementById("epoch-label").textContent = `époque ${shown} · ${acc}`;
}

// Matrice de confusion (vrai chiffre en ligne, réponse en colonne) et quelques-unes de ses erreurs.
function drawEval(ev) {
  const root = document.getElementById("train-eval");
  root.innerHTML = "";
  if (!ev) return;
  const title = document.createElement("p");
  title.className = "hint";
  title.textContent = `Sur les ${ev.n_test} chiffres jamais vus, il s'est trompé ${ev.n_wrong} fois. Ligne : le vrai chiffre ; colonne : sa réponse.`;
  root.appendChild(title);
  const grid = document.createElement("div");
  grid.className = "confusion";
  const max = Math.max(...ev.confusion.flat());
  let html = `<span></span>${[...Array(10).keys()].map((i) => `<span class="head">→${i}</span>`).join("")}`;
  ev.confusion.forEach((row, t) => {
    html += `<span class="head">${t}</span>`;
    row.forEach((n, p) => {
      const k = n / (max || 1);
      const color = t === p ? `rgba(76,217,138,${0.15 + 0.85 * k})` : `rgba(255,107,107,${n ? 0.25 + 0.75 * k : 0})`;
      html += `<span class="cell" style="background:${color}" title="${n} fois : vrai ${t}, répondu ${p}">${n || ""}</span>`;
    });
  });
  grid.innerHTML = html;
  root.appendChild(grid);
  if (!ev.mistakes.length) return;
  const gal = document.createElement("div");
  gal.className = "mistakes";
  ev.mistakes.forEach((m) => {
    const fig = document.createElement("figure");
    fig.appendChild(digitCanvas(m.image));
    const cap = document.createElement("figcaption");
    cap.textContent = `${m.true} lu ${m.pred}`;
    fig.appendChild(cap);
    fig.title = "Clique pour envoyer ce chiffre au réseau et voir pourquoi il se trompe";
    fig.addEventListener("click", () => { pixels = m.image.map((r) => r.slice()); paintPad(pixels); schedule(); });
    gal.appendChild(fig);
  });
  root.appendChild(gal);
}

function digitCanvas(img) {
  const cv = document.createElement("canvas");
  cv.width = img[0].length; cv.height = img.length;
  const ctx = cv.getContext("2d");
  const data = ctx.createImageData(cv.width, cv.height);
  img.flat().forEach((v, j) => { const g = Math.round(v * 255); data.data.set([g, g, g, 255], j * 4); });
  ctx.putImageData(data, 0, 0);
  return cv;
}

async function trainEpochs(n) {
  if (training) return;
  training = true;
  shownEpoch = null; // on suit les nouveaux poids
  const buttons = ["train-1", "train-5", "train-reset"].map((id) => document.getElementById(id));
  buttons.forEach((b) => { b.disabled = true; });
  const status = document.getElementById("train-status");
  try {
    for (let i = 0; i < n; i++) {
      status.textContent = `Époque ${i + 1} / ${n}…`;
      const r = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: getCode() }),
      });
      const res = await r.json();
      if (!res.ok) { status.textContent = res.error; return; }
      drawTraining(res.training);
      await run(); // toutes les vues avec les nouveaux poids
    }
  } catch (e) {
    status.textContent = "Le serveur ne répond pas.";
  } finally {
    training = false;
    buttons.forEach((b) => { b.disabled = false; });
    if (lastResult && lastResult.training) drawTraining(lastResult.training);
  }
}

document.getElementById("epoch-range").addEventListener("input", (e) => {
  shownEpoch = Number(e.target.value);
  schedule();
});

document.getElementById("train-1").addEventListener("click", () => trainEpochs(1));
document.getElementById("train-5").addEventListener("click", () => trainEpochs(5));
document.getElementById("train-reset").addEventListener("click", async () => {
  shownEpoch = null;
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: getCode() }),
  });
  run();
});
