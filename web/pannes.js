// Mode panne : une faute est glissée dans le code et l'on regarde à quoi elle ressemble dans les vues,
// avant de nommer sa cause. Un réseau qui n'apprend pas, un score trop beau, un agent qui tourne en
// rond ne s'annoncent jamais : l'expert les reconnaît à l'œil. Ici on apprend à les voir, puis à les
// expliquer ; la ligne fautive s'allume dans le code quand la cause est montrée.

const panne = { id: null, original: null, fini: false, vus: [] };

async function pannePost(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

async function panneStart() {
  const card = document.getElementById("panne-card");
  if (panne.id === null) panne.original = getCode(); // on garde le code sain pour « Quitter »
  const res = await pannePost("/api/panne/start", { code: panne.original, exclude: panne.vus });
  if (!res.ok) { setStatus(res.error, "err"); return; }
  Object.assign(panne, { id: res.id, fini: false });
  panne.vus.push(res.id);
  if (panne.vus.length >= res.total) panne.vus = []; // toutes vues : on recommence le tour
  card.hidden = false;
  card.classList.remove("solved");
  document.getElementById("panne-btn").classList.add("on");
  document.getElementById("panne-count").textContent = `${res.total} panne${res.total > 1 ? "s" : ""} possible${res.total > 1 ? "s" : ""} sur ce code`;
  document.getElementById("panne-symptom").textContent = `Ce qu'on voit : ${res.symptome}`;
  document.getElementById("panne-result").hidden = true;
  const choices = document.getElementById("panne-choices");
  choices.innerHTML = "";
  res.choix.forEach((c) => {
    const b = document.createElement("button");
    b.className = "small";
    b.textContent = c;
    b.addEventListener("click", () => panneGuess(c, b));
    choices.appendChild(b);
  });
  setCode(res.code); // l'éditeur relance l'analyse : les symptômes apparaissent à droite
  card.scrollIntoView({ block: "start" });
}

function panneShow(res, juste) {
  panne.fini = true;
  const out = document.getElementById("panne-result");
  out.hidden = false;
  out.className = `panne-result ${juste ? "good" : ""}`;
  const head = juste ? "<b>C'est ça.</b> " : `<b>La cause : ${res.cause}.</b> `;
  out.innerHTML = `${head}${res.explication}${res.line ? ` <span class="meta">(ligne ${res.line}, allumée dans le code)</span>` : ""}`;
  document.getElementById("panne-card").classList.add("solved");
  document.querySelectorAll("#panne-choices button").forEach((b) => {
    b.disabled = true;
    if (b.textContent === res.cause) b.classList.add("right");
  });
  if (res.line) { highlightLine(res.line); revealLine(res.line); }
}

async function panneGuess(cause, button) {
  if (panne.fini) return;
  const res = await pannePost("/api/panne/guess", { id: panne.id, cause, code: getCode() });
  if (!res.ok) { setStatus(res.error, "err"); return; }
  if (res.juste) return panneShow(res, true);
  button.classList.add("wrong");
  button.disabled = true;
  const out = document.getElementById("panne-result");
  out.hidden = false;
  out.className = "panne-result bad";
  out.textContent = "Pas cette famille-là. Regarde encore les vues, ou montre la cause.";
}

async function panneReveal() {
  if (panne.fini) return;
  const res = await pannePost("/api/panne/reveal", { id: panne.id, code: getCode() });
  if (res.ok) panneShow(res, false);
}

function panneQuit() {
  document.getElementById("panne-card").hidden = true;
  document.getElementById("panne-btn").classList.remove("on");
  if (panne.original !== null) setCode(panne.original);
  Object.assign(panne, { id: null, original: null, fini: false, vus: [] });
}

(function initPannes() {
  const btn = document.getElementById("panne-btn");
  btn.addEventListener("click", () => (panne.id === null ? panneStart() : panneQuit()));
  document.getElementById("panne-reveal").addEventListener("click", panneReveal);
  document.getElementById("panne-again").addEventListener("click", panneStart);
  document.getElementById("panne-quit").addEventListener("click", panneQuit);
  const select = document.getElementById("examples");
  const refresh = () => { btn.disabled = select.value.startsWith("Erreur"); };
  select.addEventListener("change", () => {
    // Autre exemple : on sort du mode panne sans toucher au code que l'exemple vient de charger.
    if (panne.id !== null) { panne.original = null; panneQuit(); }
    refresh();
  });
  refresh();
})();
