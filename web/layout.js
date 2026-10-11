// neuro-lens : colonnes redimensionnables. On glisse les séparateurs (explorateur | code | vues) ;
// les largeurs sont gardées d'une visite à l'autre. Double-clic : retour à la taille d'origine.
// Au clavier : Tab jusqu'au séparateur, puis flèches gauche / droite.

(() => {
  const KEY = "neuro-lens-colonnes";
  const root = document.documentElement;
  const main = document.querySelector("main");
  const codePane = document.querySelector(".code-pane");
  const explorer = document.getElementById("explorer");
  const saved = (() => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; } })();
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) { /* stockage indisponible */ } };

  // Chaque séparateur règle une variable CSS, bornée pour qu'aucune colonne ne disparaisse.
  const splitters = {
    "split-explorer": {
      name: "explorer", css: "--explorer-w",
      size: () => explorer.getBoundingClientRect().width,
      limits: () => [140, Math.max(160, codePane.getBoundingClientRect().width - 220)],
    },
    "split-main": {
      name: "code", css: "--code-w",
      size: () => codePane.getBoundingClientRect().width,
      // le code garde au moins 260 px à côté de l'explorateur, les vues au moins 340 px
      limits: () => {
        const lo = (explorer.hidden ? 0 : explorer.getBoundingClientRect().width + 7) + 260;
        return [lo, Math.max(lo + 20, main.getBoundingClientRect().width - 340)];
      },
    },
  };

  const apply = (s, width) => {
    if (width === null || width === undefined) { root.style.removeProperty(s.css); delete saved[s.name]; return; }
    const [lo, hi] = s.limits();
    const w = Math.round(Math.min(hi, Math.max(lo, width)));
    root.style.setProperty(s.css, `${w}px`);
    saved[s.name] = w;
  };

  Object.entries(splitters).forEach(([id, s]) => {
    const handle = document.getElementById(id);
    if (saved[s.name]) apply(s, saved[s.name]);
    let start = null;
    handle.addEventListener("pointerdown", (e) => {
      start = { x: e.clientX, width: s.size() };
      handle.setPointerCapture(e.pointerId);
      handle.classList.add("dragging");
      document.body.classList.add("resizing");
      e.preventDefault();
    });
    handle.addEventListener("pointermove", (e) => { if (start) apply(s, start.width + e.clientX - start.x); });
    const stop = () => {
      if (!start) return;
      start = null;
      handle.classList.remove("dragging");
      document.body.classList.remove("resizing");
      save();
    };
    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
    handle.addEventListener("dblclick", () => { apply(s, null); save(); });
    handle.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      apply(s, s.size() + (e.key === "ArrowRight" ? 24 : -24));
      save();
      e.preventDefault();
    });
  });

  // Fenêtre rétrécie : on ramène les largeurs gardées dans les bornes.
  window.addEventListener("resize", () => {
    Object.values(splitters).forEach((s) => { if (saved[s.name]) apply(s, saved[s.name]); });
  });
})();
