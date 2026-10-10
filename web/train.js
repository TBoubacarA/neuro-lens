// Carte « Entraînement » : fait apprendre le réseau époque par époque et trace la courbe d'erreur.
// Après chaque époque on relance l'analyse : toutes les vues montrent alors le réseau entraîné.

let training = false;

function renderTraining(res) {
  const card = document.getElementById("train-card");
  card.hidden = !res.trainable;
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
    return;
  }
  const last = t.epochs[t.epochs.length - 1];
  shape.textContent = `${t.epochs.length} époque${t.epochs.length > 1 ? "s" : ""}`;
  if (!training) {
    status.textContent = `${Math.round(last.test_acc * 100)} % de bonnes réponses sur des chiffres jamais vus`;
  }

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

async function trainEpochs(n) {
  if (training) return;
  training = true;
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

document.getElementById("train-1").addEventListener("click", () => trainEpochs(1));
document.getElementById("train-5").addEventListener("click", () => trainEpochs(5));
document.getElementById("train-reset").addEventListener("click", async () => {
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: getCode() }),
  });
  run();
});
