// neuro-lens : le parcours expert et le mode défi.
// Voir ne suffit pas pour apprendre : en mode défi, on prédit d'abord (formes, paramètres), l'outil
// montre ensuite. Le parcours garde des preuves datées de ce qu'on a réellement réussi
// (voir docs/recherche-experts-ia.md et server/progress.py).

let defiOn = (() => { try { return localStorage.getItem("neuro-lens-defi") === "1"; } catch (e) { return false; } })();
let defi = null; // { signature, questions, answered }

const fmtDims = (dims) => dims.join(" × ");
const one = (v) => (Array.isArray(v) ? v[0] : v);
const product = (dims) => dims.reduce((a, b) => a * b, 1);

// ---------- Mode défi ----------

function layerSource(res, l, i) {
  const from = l.inputs && l.inputs.length === 1 ? l.inputs[0] : i - 1;
  return (from < 0 ? res.input.shape : res.layers[from].shape).slice(1);
}

function describeLayer(l) {
  const p = l.params;
  if (l.type === "Conv2d") return `Conv2d : ${p.in_channels} → ${p.out_channels} canaux, noyau ${one(p.kernel_size)}, pas ${one(p.stride)}, marge ${one(p.padding)}`;
  if (l.type === "MaxPool2d" || l.type === "AvgPool2d") return `${l.type} : carrés de ${one(p.kernel_size)}`;
  if (l.type === "Linear") return `Linear : ${p.in_features} → ${p.out_features}`;
  if (l.type === "Embedding") return `Embedding : table de ${p.num_embeddings} lignes, vecteurs de ${p.embedding_dim}`;
  return l.type;
}

function whyShape(l, from) {
  const p = l.params, out = l.shape.slice(1);
  if (l.type === "Conv2d" && from.length === 3) {
    const k = one(p.kernel_size), s = one(p.stride), pad = one(p.padding);
    return `${p.out_channels} filtres donnent ${p.out_channels} canaux. Côté : (${from[1]} + 2 × ${pad} − ${k}) / ${s} + 1 = ${out[1]}.`;
  }
  if ((l.type === "MaxPool2d" || l.type === "AvgPool2d") && from.length === 3) {
    return `Les canaux ne changent pas. Côté : ${from[1]} / ${one(p.kernel_size)} = ${out[1]}.`;
  }
  if (l.type === "Flatten") return `Tout est mis bout à bout : ${fmtDims(from)} = ${product(from)} valeurs.`;
  if (l.type === "Linear") return `Un nombre par neurone : ${p.out_features} neurones, donc ${p.out_features} valeurs${out.length > 1 ? " par token" : ""}.`;
  if (l.type === "Embedding") return `Chaque token devient un vecteur de ${p.embedding_dim} nombres.`;
  return "";
}

function whyParams(l) {
  const p = l.params;
  if (l.type === "Conv2d") {
    const k = one(p.kernel_size), w = k * k * p.in_channels * p.out_channels;
    if (w + p.out_channels === l.n_params) return `${k} × ${k} × ${p.in_channels} × ${p.out_channels} poids + ${p.out_channels} biais = ${l.n_params}.`;
  }
  if (l.type === "Linear" && p.in_features * p.out_features + p.out_features === l.n_params) {
    return `${p.in_features} × ${p.out_features} poids + ${p.out_features} biais = ${l.n_params}.`;
  }
  return "";
}

function buildQuestions(res) {
  const changing = res.layers.map((l, i) => ({ l, i, from: layerSource(res, l, i) }))
    .filter(({ l, from }) => JSON.stringify(from) !== JSON.stringify(l.shape.slice(1)));
  // trois couches au plus, réparties du début à la fin du réseau
  const picks = changing.length <= 3 ? changing
    : [changing[0], changing[Math.floor(changing.length / 2)], changing[changing.length - 1]];
  const questions = picks.map(({ l, i, from }) => ({
    kind: "shape", index: i, line: l.line, title: l.type, // les réglages se lisent dans le code, pas ici
    ask: `Elle reçoit ${fmtDims(from)}. Quelle forme en sort ?`,
    expected: l.shape.slice(1), why: whyShape(l, from), hint: "ex. 8 × 14 × 14",
  }));
  const counted = res.layers.map((l, i) => ({ l, i })).filter(({ l }) => (l.type === "Conv2d" || l.type === "Linear") && l.n_params);
  if (counted.length) {
    const { l, i } = counted.find(({ l }) => l.type === "Conv2d") ?? counted[0]; // la moins évidente à compter
    questions.push({
      kind: "params", index: i, line: l.line, title: describeLayer(l),
      ask: "Combien de paramètres cette couche apprend-elle (poids et biais) ?",
      expected: [l.n_params], why: whyParams(l), hint: "un nombre",
    });
  }
  return questions;
}

// Appelé après chaque rendu : `res` est l'analyse d'un réseau PyTorch, ou null pour le reste.
function defiUpdate(res) {
  const card = $("defi-card"), lens = $("lens");
  $("defi-btn").classList.toggle("on", defiOn);
  $("defi-btn").textContent = defiOn ? "Mode défi : activé" : "Mode défi";
  if (!defiOn || !res) { defi = null; card.hidden = true; lens.classList.remove("defi-masque"); return; }
  const signature = res.layers.map((l) => `${l.type}:${l.shape.join("x")}`).join("|");
  if (!defi || defi.signature !== signature) {
    const questions = buildQuestions(res);
    defi = questions.length ? { signature, questions, answered: false } : null;
  }
  if (!defi) { card.hidden = true; lens.classList.remove("defi-masque"); return; }
  lens.classList.toggle("defi-masque", !defi.answered);
  if (!defi.answered) setStatus("mode défi : prédis d'abord", "");
  drawDefi();
}

function drawDefi() {
  const card = $("defi-card");
  card.hidden = false;
  const kept = [...card.querySelectorAll("input")].map((x) => x.value); // ce qui est déjà tapé survit à un nouveau rendu
  card.innerHTML = `<div class="card-head"><span class="tag tag-input">Défi : prédis avant de voir</span><span class="shape"></span></div>
    <p class="explain"></p><div class="defi-list"></div><div class="train-actions"></div><p class="meta defi-result"></p>`;
  card.querySelector(".explain").textContent = defi.answered
    ? "Voici les réponses, avec le calcul. Les vues du réseau sont affichées en dessous."
    : "Les vues sont cachées. Lis ton code, calcule de tête ou sur papier, réponds, puis vérifie. Se tromper ici fait partie de l'exercice : c'est l'écart entre ta prédiction et la réalité qui fait apprendre.";
  const list = card.querySelector(".defi-list");
  defi.questions.forEach((q, n) => {
    const row = document.createElement("div");
    row.className = "defi-row";
    row.innerHTML = `<b></b><span></span><input type="text" inputmode="numeric" autocomplete="off"><p class="meta"></p>`;
    row.querySelector("b").textContent = `Couche ${q.index + 1}${q.line ? ` · ligne ${q.line}` : ""} — ${q.title}`;
    row.querySelector("span").textContent = q.ask;
    const input = row.querySelector("input");
    input.placeholder = q.hint;
    input.value = q.given ?? kept[n] ?? "";
    input.disabled = defi.answered;
    input.setAttribute("aria-label", q.ask);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") checkDefi(); });
    row.addEventListener("mouseenter", () => highlightLine(q.line));
    row.addEventListener("mouseleave", () => highlightLine(null));
    if (defi.answered) {
      row.classList.add(q.right ? "right" : "wrong");
      row.querySelector("p").textContent = `${q.right ? "✓ Juste" : "✗ Non"} : ${q.kind === "params" ? q.expected[0] : fmtDims(q.expected)}. ${q.why}`;
    }
    list.appendChild(row);
  });
  const actions = card.querySelector(".train-actions");
  if (!defi.answered) {
    const go = document.createElement("button");
    go.textContent = "Vérifier";
    go.addEventListener("click", checkDefi);
    const skip = document.createElement("button");
    skip.className = "small";
    skip.textContent = "Voir sans répondre";
    skip.title = "Affiche les vues ; ce défi ne comptera pas";
    skip.addEventListener("click", () => { defi.answered = true; defi.skipped = true; defi.questions.forEach((q) => { q.right = false; q.given = ""; }); finishDefi(); });
    actions.append(go, skip);
  }
  if (defi.answered) {
    const right = defi.questions.filter((q) => q.right).length;
    card.querySelector(".shape").textContent = defi.skipped ? "passé" : `${right} / ${defi.questions.length}`;
    card.querySelector(".defi-result").textContent = defi.skipped ? "Défi passé : rien n'est noté dans ton parcours."
      : right === defi.questions.length ? "Sans faute : c'est noté comme preuve dans ton parcours (niveau 1). Change le réseau pour un nouveau défi."
      : "Pas encore un sans-faute : relis le calcul des lignes en rouge, puis change le réseau pour réessayer. Seul un sans-faute compte comme preuve.";
  }
}

async function checkDefi() {
  if (!defi || defi.answered) return;
  const inputs = [...$("defi-card").querySelectorAll("input")];
  defi.questions.forEach((q, n) => {
    q.given = inputs[n].value;
    const digits = q.kind === "params" ? [Number(q.given.replace(/\D/g, ""))] : (q.given.match(/\d+/g) || []).map(Number);
    q.right = q.given.trim() !== "" && JSON.stringify(digits) === JSON.stringify(q.expected);
  });
  defi.answered = true;
  const right = defi.questions.filter((q) => q.right).length;
  try {
    await fetch("/api/progress/challenge", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signature: defi.signature, right, total: defi.questions.length }) });
  } catch (e) { /* le défi reste corrigé même si la preuve n'a pas pu être notée */ }
  finishDefi();
}

function finishDefi() {
  $("lens").classList.remove("defi-masque");
  if (lastResult) setStatus(`${lastResult.layers.length} couches · ${fmtN(lastResult.n_params)} paramètres`, "ok");
  drawDefi();
  if (!$("parcours-card").hidden) loadParcours();
}

// ---------- Parcours expert ----------

const CHECK_ICON = { done: "✓", confirm: "◐", doing: "…", todo: "○", untracked: "—" };
const LEVEL_TEXT = { validated: "validé", measured: "mesuré", started: "en cours", todo: "à faire", empty: "à venir" };

async function loadParcours() {
  let data;
  try { data = await (await fetch("/api/progress")).json(); } catch (e) { return; }
  const box = $("parcours-levels");
  box.innerHTML = "";
  data.levels.forEach((level) => {
    const row = document.createElement("details");
    row.className = `level ${level.status}`;
    row.open = level.status === "started" || level.id === 1;
    row.innerHTML = `<summary><i></i><b></b><span class="level-status"></span></summary><p class="meta"></p><ul></ul>`;
    row.querySelector("i").textContent = level.id;
    row.querySelector("b").textContent = level.name;
    row.querySelector(".level-status").textContent = LEVEL_TEXT[level.status] + (level.tracked ? ` · ${level.done}/${level.tracked}` : "");
    row.querySelector("p").textContent = level.goal;
    level.checks.forEach((c) => {
      const li = document.createElement("li");
      li.className = c.status;
      li.innerHTML = `<i></i><div><span></span><small></small></div>`;
      li.querySelector("i").textContent = CHECK_ICON[c.status];
      li.querySelector("span").textContent = c.label;
      li.querySelector("small").textContent =
        c.status === "untracked" ? "pas encore vérifié par l'outil"
        : c.status === "done" ? `${c.count} preuve${c.count > 1 ? "s" : ""}, dernière le ${c.proofs[c.proofs.length - 1].at}`
        : c.status === "confirm" ? `${c.count} preuves le même jour : refais-en une un autre jour pour confirmer (réussir à chaud ne prouve pas qu'on a retenu)`
        : `${c.count} / ${c.need} · ${c.how}`;
      row.querySelector("ul").appendChild(li);
    });
    box.appendChild(row);
  });
}

(() => {
  $("parcours-btn").addEventListener("click", () => {
    const card = $("parcours-card");
    card.hidden = !card.hidden;
    $("parcours-btn").classList.toggle("on", !card.hidden);
    if (!card.hidden) { loadParcours(); $("lens").scrollTop = 0; }
  });
  $("defi-btn").addEventListener("click", () => {
    defiOn = !defiOn;
    try { localStorage.setItem("neuro-lens-defi", defiOn ? "1" : "0"); } catch (e) { /* stockage indisponible */ }
    defiUpdate(otherKind ? null : lastResult);
    if (!defiOn && lastResult && !otherKind) setStatus(`${lastResult.layers.length} couches · ${fmtN(lastResult.n_params)} paramètres`, "ok");
  });
  $("defi-btn").classList.toggle("on", defiOn);
  $("defi-btn").textContent = defiOn ? "Mode défi : activé" : "Mode défi";
})();
