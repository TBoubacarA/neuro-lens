// Carte « Ce que le réseau imagine » : pour chaque chiffre, l'image qui convainc le plus le réseau.

document.getElementById("dream-go").addEventListener("click", async () => {
  const button = document.getElementById("dream-go");
  const status = document.getElementById("dream-status");
  button.disabled = true;
  status.textContent = "Le réseau imagine…";
  try {
    const r = await fetch("/api/dream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: getCode() }),
    });
    const res = await r.json();
    if (!res.ok) { status.textContent = res.error; return; }
    status.textContent = "";
    document.getElementById("dream-shape").textContent =
      res.epochs ? `après ${res.epochs} époque${res.epochs > 1 ? "s" : ""}` : "poids aléatoires";
    const root = document.getElementById("dreams");
    root.innerHTML = "";
    res.images.forEach((img, digit) => {
      const fig = document.createElement("figure");
      const cv = document.createElement("canvas");
      cv.width = img[0].length;
      cv.height = img.length;
      const ctx = cv.getContext("2d");
      const data = ctx.createImageData(cv.width, cv.height);
      img.flat().forEach((v, i) => data.data.set([v * 255, v * 255, v * 255, 255], i * 4));
      ctx.putImageData(data, 0, 0);
      fig.appendChild(cv);
      const cap = document.createElement("figcaption");
      cap.innerHTML = `<b>${digit}</b> sûr à ${Math.round(res.confidence[digit] * 100)} %`;
      fig.appendChild(cap);
      root.appendChild(fig);
    });
  } catch (e) {
    status.textContent = "Le serveur ne répond pas.";
  } finally {
    button.disabled = false;
  }
});
