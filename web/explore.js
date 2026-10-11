// Cartes « Tromper le réseau » (attaque adverse) et « Paysage de l'erreur ».

function renderExplore(res) {
  const can = Boolean(res.trainable);
  document.getElementById("attack-card").hidden = !can;
  document.getElementById("landscape-card").hidden = !can;
  if (!can) return;
  const sel = document.getElementById("attack-target");
  if (!sel.options.length) for (let i = 0; i < 10; i++) sel.add(new Option(`« ${i} »`, i));
  if (res.prediction && Number(sel.value) === res.prediction.top) sel.value = (res.prediction.top + 1) % 10;
  document.getElementById("attack-shape").textContent = res.prediction ? `il lit ${res.prediction.top}` : "";
  const t = res.training;
  document.getElementById("landscape-shape").textContent =
    t && t.epochs.length ? `${t.epochs.length} époque${t.epochs.length > 1 ? "s" : ""}` : "poids de départ";
}

function grayCanvas(img, signed = false) {
  const cv = document.createElement("canvas");
  cv.width = img[0].length; cv.height = img.length;
  const ctx = cv.getContext("2d");
  const data = ctx.createImageData(cv.width, cv.height);
  img.flat().forEach((v, j) => {
    if (signed) { // rouge = pixel assombri, vert = éclairci
      const a = Math.min(1, Math.abs(v));
      data.data.set(v >= 0 ? [40, 40 + 180 * a, 60, 255] : [40 + 215 * a, 40, 60, 255], j * 4);
    } else {
      const g = Math.round(Math.min(1, Math.max(0, v)) * 255);
      data.data.set([g, g, g, 255], j * 4);
    }
  });
  ctx.putImageData(data, 0, 0);
  return cv;
}

function probsLine(p) {
  const best = p.top;
  return `lit <b>${best}</b> (sûr à ${Math.round(p.probs[best] * 100)} %)`;
}

document.getElementById("attack-go").addEventListener("click", async () => {
  const status = document.getElementById("attack-status");
  const root = document.getElementById("attack-result");
  const target = Number(document.getElementById("attack-target").value);
  status.textContent = "Je cherche la retouche…";
  try {
    const r = await fetch("/api/attack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: getCode(), pixels, target }),
    });
    const res = await r.json();
    if (!res.ok) { status.textContent = res.error; return; }
    status.textContent = "";
    root.innerHTML = "";
    const col = (title, cv, caption) => {
      const fig = document.createElement("figure");
      fig.appendChild(cv);
      fig.insertAdjacentHTML("beforeend", `<figcaption><b>${title}</b><br>${caption}</figcaption>`);
      root.appendChild(fig);
      return fig;
    };
    col("Ton dessin", grayCanvas(res.original), probsLine(res.before));
    col("La retouche", grayCanvas(res.delta, true), `chaque pixel bouge d'au plus ${Math.round(res.max_change * 100)} %`);
    const after = col("Après", grayCanvas(res.image), probsLine(res.after));
    after.title = "Clique pour envoyer cette image au réseau et voir toutes les vues";
    after.style.cursor = "pointer";
    after.addEventListener("click", () => { pixels = res.image.map((row) => row.slice()); paintPad(pixels); schedule(); });
    const verdict = document.createElement("p");
    verdict.className = "hint";
    verdict.textContent = res.fooled
      ? `Trompé : avec une retouche que ton œil ne voit presque pas, il lit ${res.after.top} au lieu de ${res.before.top}. Clique sur « Après » pour voir ce que les couches en font.`
      : `Pas trompé cette fois : il lit encore ${res.after.top}. Entraîne-le davantage, ou vise un autre chiffre.`;
    root.appendChild(verdict);
  } catch (e) {
    status.textContent = "Le serveur ne répond pas.";
  }
});

document.getElementById("landscape-go").addEventListener("click", async () => {
  const status = document.getElementById("landscape-status");
  const root = document.getElementById("landscape");
  status.textContent = "Je mesure l'erreur tout autour…";
  try {
    const r = await fetch("/api/landscape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: getCode() }),
    });
    const res = await r.json();
    if (!res.ok) { status.textContent = res.error; return; }
    status.textContent = res.pca
      ? `Axes : les deux directions où les poids ont le plus bougé. Mesuré sur ${res.n_samples} chiffres.`
      : `Pas encore de chemin : entraîne-le pour voir la descente. Mesuré sur ${res.n_samples} chiffres.`;
    drawLandscape(root, res);
  } catch (e) {
    status.textContent = "Le serveur ne répond pas.";
  }
});

function drawLandscape(root, res) {
  const N = res.grid.length, S = 300;
  const flat = res.grid.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const cv = document.createElement("canvas");
  cv.width = N; cv.height = N;
  const ctx = cv.getContext("2d");
  const data = ctx.createImageData(N, N);
  flat.forEach((v, j) => {
    const t = Math.log1p(v - lo) / (Math.log1p(hi - lo) || 1); // échelle log : le fond de la vallée reste lisible
    data.data.set(heat(t), j * 4);
  });
  ctx.putImageData(data, 0, 0);
  const toPx = (u) => ((u + res.radius) / (2 * res.radius)) * S;
  let path = "";
  if (res.path.length) {
    path += `<polyline points="${res.path.map(([u, v]) => `${toPx(u)},${toPx(v)}`).join(" ")}"/>`;
    res.path.forEach(([u, v, k], i) => {
      const last = i === res.path.length - 1;
      path += `<circle cx="${toPx(u)}" cy="${toPx(v)}" r="${last ? 6 : 3.5}" class="${last ? "now" : ""}"><title>époque ${k}</title></circle>`;
      if (i === 0 || last) path += `<text x="${toPx(u) + 8}" y="${toPx(v) - 6}">${i === 0 ? "départ" : "maintenant"}</text>`;
    });
  } else {
    path += `<circle cx="${S / 2}" cy="${S / 2}" r="6" class="now"><title>poids actuels</title></circle>`;
  }
  root.innerHTML = `
    <div class="landscape-box" style="width:${S}px;height:${S}px">
      <img alt="paysage de l'erreur" src="${cv.toDataURL()}" style="width:${S}px;height:${S}px">
      <svg viewBox="0 0 ${S} ${S}" style="width:${S}px;height:${S}px">${path}</svg>
    </div>
    <p class="meta">erreur de ${lo.toFixed(2)} (sombre) à ${hi.toFixed(2)} (clair)</p>`;
}
