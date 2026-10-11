# Former des experts IA avec neuro-lens : recherche et feuille de route

*Recherche du 12 octobre 2026. Sources web consultées listées en fin de document. Ce qui n'a pas pu être vérifié sur la source d'origine est marqué **[non vérifié]**.*

## Positionnement (décision du propriétaire, 12 octobre 2026)

neuro-lens **n'est pas une plateforme d'exercices notés** et ne cherche pas à concurrencer les outils de ce genre. Ce qui le distingue :

1. **La visualisation, y compris en 3D**, de ce qu'on développe, sur son propre code.
2. **La vulgarisation** : chaque élément expliqué en langage simple, avec les valeurs réelles du projet, pour comprendre toute la chaîne jusqu'en production.
3. **L'architecture de chaque type de projet** : données → modèle → infrastructure → production.

Conséquences sur ce document :

- L'ordre de construction de la section 5 est révisé ci-dessous. Le mode défi reste une option, désactivée par défaut ; aucune autre mécanique de quiz, de score ou de classement ne sera ajoutée.
- La recherche déconseillait d'ajouter des vues 3D (section 3, « À ne pas construire »). Le propriétaire en a décidé autrement : la 3D est la signature de l'outil. Règle retenue pour qu'elle reste utile : **en 3D, chaque dimension porte une information** (dans la carte de la chaîne, la hauteur d'une brique dit si elle existe).
- Ce qui reste valable de la section 2 : faire comprendre en montrant l'état réel, sans rien de caché, et expliquer chaque chose isolément avant de les combiner.

**Ordre de construction révisé**

1. **Carte 3D de la chaîne** par type de projet (fait : ML classique, réseau de neurones, agent).
2. **Poser les briques manquantes** de cette carte, en commençant par la chaîne de production des réseaux PyTorch (version figée, image, API, surveillance) et le réentraînement.
3. **Architectures de référence** des autres types de projet : application LLM avec recherche documentaire (RAG), ajustement d'un modèle, traitement par lots.
4. **Santé de l'entraînement couche par couche** et **transformer token par token** (capacités 1 et 3), en restant sur le principe « voir pour comprendre ».

## En bref

1. **Un expert se distingue moins par le modèle que par ce qui l'entoure.** Tous les référentiels consultés (Google, Microsoft, Anthropic, OpenAI, OWASP) convergent : métriques définies avant le modèle, tests des données et de l'infrastructure, même code à l'entraînement et au service, surveillance, évaluations.
2. **Voir ne suffit pas pour apprendre.** neuro-lens montre déjà très bien ; la recherche sur l'apprentissage dit qu'il faut en plus faire *prédire*, faire *rappeler*, faire *réparer*, et retirer l'aide à mesure que l'ingénieur progresse. C'est le plus gros manque de l'outil aujourd'hui.
3. **Douze capacités manquent**, classées plus bas. Les trois premières : la santé de l'entraînement couche par couche, le graphe de rétropropagation, la passe avant d'un transformer token par token.
4. **Proposition : un parcours expert en 8 niveaux**, où chaque niveau est validé par une vérification automatique et non par une lecture.

## 1. Ce qu'un ingénieur IA expert maîtrise

| Domaine | Ce qui sépare l'expert du débutant | Piège typique du débutant |
|---|---|---|
| **Données** | Valider schéma et distributions avant l'entraînement ; découper sans fuite (temporel, par entité) ; versionner données et étiquettes ; tester les données comme du code | Fuite de données, découpage aléatoire sur des données temporelles : score flatteur, non reproductible |
| **Entraînement** | Commencer par un modèle simple ; établir une ligne de base ; suivre les expériences ; reproductibilité (graines, environnement) ; diagnostiquer une courbe de perte | Modèle complexe d'emblée, sans ligne de base ni suivi |
| **Évaluation** | Définir les métriques avant le modèle ; évaluer par segments ; comparer au modèle en production ; seuils bloquants avant export | Une seule métrique agrégée, choisie après coup |
| **Service** | Même code à l'entraînement et au service ; registre avec retour arrière ; tests de l'API ; déploiement progressif | Script de prédiction écrit à la main, non versionné |
| **Surveillance** | Mesurer l'écart entraînement/service ; dérive des données et dérive du concept ; pannes silencieuses ; déclencheurs de réentraînement | Surveiller le temps de réponse, pas la qualité des réponses |
| **Applications LLM** | Écrire les évaluations avant d'itérer sur le prompt ; cas tirés du trafic réel ; calibrer un juge automatique sur des étiquettes humaines ; versionner les prompts ; en RAG, évaluer séparément la recherche et la génération | « Ça a l'air de marcher » ; évaluations ajoutées après la mise en production |
| **Agents** | Préférer un enchaînement fixe à un agent quand la tâche est bien définie ; maîtriser les 5 motifs (chaînage, routage, parallélisation, orchestrateur-ouvriers, évaluateur-optimiseur) ; concevoir les outils comme une interface ; noter le résultat et non le chemin ; lire les transcriptions ; encadrer les actions d'écriture | Agent là où un appel suffit ; aucune évaluation ; erreurs qui s'accumulent d'étape en étape |
| **Infrastructure** | Conteneurs reproductibles ; intégration continue qui teste code, données et modèle ; déploiement par paliers ; coût (jetons, GPU) suivi comme une métrique | Une CI qui ne teste que le code applicatif |
| **Architecture du code** | Limiter l'enchevêtrement (« changer une chose change tout ») ; éviter le code-colle ; traiter la configuration comme du code ; tester l'infrastructure indépendamment du modèle | Notebook monolithique, configuration dispersée |
| **Sécurité et gouvernance** | OWASP LLM 2025 (injection de prompt, sortie mal traitée, agentivité excessive, consommation non bornée…) ; NIST AI RMF ; obligations de l'AI Act pour le haut risque (journalisation, documentation, supervision humaine) | Faire confiance à la sortie du modèle et lui donner trop de permissions |

Échelles de maturité utilisables comme repères : Google MLOps (0 manuel, 1 pipeline automatisé, 2 CI/CD du pipeline) ; Microsoft MLOps (0 à 4, de « pas de MLOps » à « opérations entièrement automatisées »).

**Où en est neuro-lens** : la chaîne *construire* est couverte pour scikit-learn (essais, versions, service, image, API, dérive) et en partie pour les agents (trace, coût, garde-fous, évaluation). Restent peu ou pas couverts : la validation des données en amont, l'évaluation par segments, la parité entraînement/service, le RAG, la sécurité des agents, et toute la chaîne de production pour les réseaux PyTorch.

## 2. Comment on devient expert, et ce que ça implique pour l'outil

| Principe | Ce que dit la recherche | Solidité | Ce que neuro-lens devrait faire |
|---|---|---|---|
| **Prédire puis observer** | Faire prédire et justifier avant de montrer révèle les idées fausses (White et Gunstone, 1992) | Validé en enseignement des sciences ; pas d'étude contrôlée trouvée en programmation | « Prédis la forme de sortie, le nombre de paramètres, le score après dérive » avant d'afficher, puis mesurer l'écart |
| **Rappel actif** | Se tester fait mieux retenir à une semaine que relire, alors que relire donne l'impression de mieux savoir (Roediger et Karpicke, 2006) | Solide | Questions de rappel sans le code sous les yeux, avant de rouvrir un projet |
| **Répétition espacée** | L'intervalle optimal entre deux révisions croît avec la durée de rétention visée (Cepeda et al., 2006, 317 expériences) | Solide, surtout sur du rappel verbal | Reproposer un défi réussi à J+2, J+7, J+30 ; une compétence non revue pâlit sur la carte |
| **Difficultés désirables** | Ce qui freine la performance immédiate (espacer, mélanger, se tester) améliore la rétention ; réussir en séance ne prouve pas qu'on a appris (Bjork, 1994) | Solide | Mélanger les types de défis ; ne jamais valider une compétence sur une seule réussite à chaud |
| **Échec productif** | Chercher avant de recevoir l'explication améliore la compréhension (Sinha et Kapur, 2021, 53 études) | Taille d'effet **[non vérifiée]** | Mode « essaie d'abord » : le défi précède l'explication |
| **Guidage dégressif** | Ce qui aide un novice gêne un avancé (effet d'inversion de l'expertise, Kalyuga et al., 2003) | Résumé lu, pas l'article | Projet commenté, puis à trous, puis page blanche ; aides masquables |
| **Réparer** | Apprendre sur des exemples erronés peut aider, surtout les plus expérimentés | Faible : peu d'essais contrôlés, et déboguer sans méthode ne marche pas | Projets cassés à réparer, avec une méthode de diagnostic explicite et des indices gradués |
| **Maîtrise** | Exiger la maîtrise avant d'avancer améliore les résultats (Kulik et al., 1990, 108 évaluations) | Effet chiffré **[source secondaire]** | Carte de compétences où chaque case renvoie à une preuve datée |
| **Retour immédiat, état visible** | Montrer le flux et l'état, sans rien de caché (Bret Victor, 2012) | Essai de conception, pas une étude | C'est le cœur actuel de l'outil : l'étendre à l'exécution pas à pas |
| **Pratique délibérée** | L'effort ciblé sur ses faiblesses, avec retour, fait progresser (Ericsson et al., 1993) | Contestée : une méta-analyse de 2014 lui attribue moins de 1 % de la variance dans les professions | Proposer l'exercice sur la compétence la plus faible, pas la suivante dans l'ordre |

**Deux règles à en tirer** : une compétence n'est acquise qu'après une réussite *différée* ; l'aide diminue de niveau en niveau.

## 3. Ce que font les outils existants et que neuro-lens n'a pas

Classement par valeur pour un ingénieur qui veut devenir expert.

| # | Capacité manquante | Pourquoi elle compte | Qui prouve que ça marche | Piste de réalisation |
|---|---|---|---|---|
| 1 | **Santé de l'entraînement couche par couche** : histogrammes d'activations et de gradients, neurones morts ou saturés, rapport mise à jour / poids | Distingue « la perte descend » de « je sais pourquoi elle ne descend pas » | TensorBoard ; cours de Karpathy (leçon 4) | Crochets PyTorch, statistiques par époque, petits histogrammes sur chaque carte de couche |
| 2 | **Graphe de rétropropagation** avec valeur et gradient par nœud | Rend le calcul du gradient tangible sur son propre code | micrograd (`draw_dot`) | Parcourir `loss.grad_fn` sur un petit modèle ; lecture pas à pas en sens inverse |
| 3 | **Passe avant d'un transformer token par token** : tokens, scores, effet de la température, du top-k, du top-p ; ce que « pense » chaque couche (logit lens) | C'est le fonctionnement réel d'un LLM, que la plupart des utilisateurs n'ont jamais vu | Transformer Explainer ; Ecco | Le serveur renvoie les meilleurs scores, le rééchantillonnage se fait dans la page |
| 4 | **Remplacement d'activation** (activation patching) | Généralise « éteindre un neurone » : méthode causale de base de l'interprétabilité | TransformerLens | Deux entrées (propre / corrompue), carte de chaleur couche × position |
| 5 | **RAG visible** : découpage, scores de recherche, passages retenus, métriques séparées recherche / génération | Cas d'usage le plus courant des LLM en entreprise ; absent de l'outil | Ragas ; Phoenix | Index local, passages surlignés dans la réponse |
| 6 | **Matrice d'évaluation et calibration du juge** : prompts × modèles × cas, accord entre le juge automatique et une annotation humaine | Un juge non calibré donne de faux scores | promptfoo ; Langfuse ; Inspect | Étendre l'évaluation actuelle en tableau, un bouton d'annotation par case |
| 7 | **Voyage dans le temps avec bifurcation** pour les agents : modifier l'état à une étape et repartir | Le rejeu montre ce qui s'est passé ; la bifurcation montre ce qui se serait passé | LangSmith Studio ; Phoenix | Points de reprise LangGraph, arbre des branches ; export des traces au format OpenTelemetry GenAI |
| 8 | **Analyse d'erreurs par tranche et calibration** | Un score global masque les défaillances | Evidently | Matrice de confusion cliquable, métriques par sous-groupe, courbe de fiabilité |
| 9 | **Balayage d'hyperparamètres** | Le curseur actuel règle un paramètre ; les interactions entre plusieurs restent invisibles | Weights & Biases | Grille en tâche de fond, coordonnées parallèles reliées au suivi des essais |
| 10 | **Contrats de données et versionnage** | La dérive est surveillée en aval, rien ne valide les données en amont | Great Expectations ; DVC | Assertions déclaratives sur les colonnes, rapport, étapes en graphe |
| 11 | **Dynamique des optimiseurs** : SGD, momentum, Adam superposés sur le paysage de l'erreur | Comprendre pourquoi un réglage diverge | Distill, « Why Momentum Really Works » | Rejouer un entraînement court avec plusieurs optimiseurs |
| 12 | **Dictionnaire de caractéristiques et graphe d'attribution** | État de l'art de l'interprétabilité ; coût élevé | circuit-tracer ; Neuronpedia | D'abord un petit autoencodeur parcimonieux sur un modèle jouet |

**À ne pas construire** : un lecteur de formats de modèles (Netron le fait) ; un tableau de bord d'observabilité de production (Langfuse, Phoenix et MLflow s'hébergent soi-même : mieux vaut exporter vers eux) ; davantage de vues 3D ; un moteur de carnet réactif (marimo) ; GAN et diffusion à taille réelle.

## 4. Proposition : un parcours expert en 8 niveaux

Chaque niveau est validé par une vérification automatique, pas par une lecture.

| Niveau | Ce que l'ingénieur construit | Ce qu'il doit savoir expliquer | Vérification automatique |
|---|---|---|---|
| **1. Lire un modèle** | Un petit réseau dont il prédit les formes et le nombre de paramètres | Pourquoi chaque couche a cette forme | Ses prédictions comparées aux valeurs calculées |
| **2. Entraîner proprement** | Ligne de base, découpage sans fuite, suivi des essais | Pourquoi ce découpage, pourquoi cette métrique | Test caché qui détecte une fuite ; au moins 3 essais suivis ; même résultat à graine fixe |
| **3. Tester et évaluer** | Tests du code, des données et du modèle ; évaluation par segments | Ce que chaque test protège ; où le modèle échoue | Défauts injectés (schéma changé, valeurs manquantes) que ses tests doivent attraper |
| **4. Servir** | Registre, service, image, API | Comment la parité entraînement/service est garantie ; comment revenir en arrière | Conteneur construit et lancé ; test de contrat de l'API ; retour arrière exécuté |
| **5. Surveiller** | Détection de dérive et règle de réentraînement | Dérive des données contre dérive du concept | Dérive injectée à l'aveugle : l'alerte doit se lever, sans fausse alerte sur un lot sain |
| **6. Application LLM évaluée** | Prompt versionné, RAG, 20 à 50 cas d'évaluation | Pourquoi ces cas ; accord du juge avec ses propres étiquettes | Diversité des cas ; taux d'accord ; régression bloquée en CI |
| **7. Agent sûr et observable** | Agent avec outils, trace, budget, garde-fous | Enchaînement fixe ou agent, et pourquoi | Attaques injectées (injection de prompt, outil qui ment, boucle) ; plafond de coût respecté |
| **8. Système complet et incident** | Projet de bout en bout, puis un projet cassé inconnu à réparer | Post-mortem : cause, détection, prévention | CI verte ; panne réparée en temps borné ; rappel des niveaux 1 à 7 réussi à J+30 |

## 5. Ordre de construction recommandé

1. **Mode défi « prédis avant de voir »** (niveau 1) et **carte de compétences avec preuves** : c'est ce qui transforme l'outil de démonstration en outil d'entraînement, pour un coût faible.
2. **Santé de l'entraînement couche par couche** (capacité 1) : le diagnostic le plus utile au quotidien.
3. **Projets cassés à réparer** (niveaux 2, 3, 5) : fuite de données, test manquant, dérive non détectée.
4. **Passe avant d'un transformer token par token** (capacité 3), puis **RAG visible** (capacité 5).
5. **Chaîne de production pour les réseaux PyTorch**, puis **sécurité des agents** (niveau 7).

## Limites de cette recherche

- Les colonnes « manque » du paysage d'outils reposent sur la description de neuro-lens, pas sur une comparaison outil par outil en conditions réelles.
- Non consultés directement : les livres de Chip Huyen (*Designing Machine Learning Systems*, *AI Engineering*), le modèle de maturité AWS, roadmap.sh (page non chargée), nanoGPT, torchinfo, Jupyter widgets, Streamlit, Gradio.
- Plusieurs tailles d'effet en science de l'apprentissage viennent de sources secondaires ; elles sont marquées dans le tableau.
- Les conventions OpenTelemetry pour l'IA générative sont encore au statut « Development », donc instables.

## Sources consultées

**Compétences et pratiques**
- Google Cloud, *MLOps: Continuous delivery and automation pipelines in machine learning* — https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- Google, *Rules of Machine Learning* — https://developers.google.com/machine-learning/guides/rules-of-ml
- Breck et al., *The ML Test Score* — https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems* — https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
- Microsoft, modèles de maturité MLOps et GenAIOps — https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model ; https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-llmops-maturity
- Anthropic, *Building effective agents* et *Demystifying evals for AI agents* — https://www.anthropic.com/engineering/building-effective-agents ; https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- OpenAI, *Evaluation best practices* — https://developers.openai.com/api/docs/guides/evaluation-best-practices
- Chip Huyen, *Agents* — https://huyenchip.com/2025/01/07/agents.html
- OWASP, *Top 10 for LLM Applications* — https://genai.owasp.org/llm-top-10/
- NIST, *AI Risk Management Framework* — https://www.nist.gov/itl/ai-risk-management-framework
- AI Act, résumé — https://artificialintelligenceact.eu/high-level-summary/
- OpenTelemetry, conventions GenAI — https://github.com/open-telemetry/semantic-conventions-genai
- Made With ML — https://madewithml.com/ ; Full Stack Deep Learning — https://fullstackdeeplearning.com/course/2022/

**Science de l'apprentissage**
- Bret Victor, *Learnable Programming* — https://worrydream.com/LearnableProgramming/
- Nicky Case, *Explorable Explanations* — https://blog.ncase.me/explorable-explanations/
- Bjork Lab (UCLA) — https://bjorklab.psych.ucla.edu/research/
- Roediger et Karpicke (2006) — https://www.psychologicalscience.org/journals/psychological-science/j.1467-9280.2006.01693.x/
- Cepeda et al. (2006) — https://pubmed.ncbi.nlm.nih.gov/16719566/
- Kalyuga et al. (2003) — https://ro.uow.edu.au/articles/journal_contribution/The_expertise_reversal_effect/27724950/1
- Pratique délibérée, réplication et méta-analyse — https://pmc.ncbi.nlm.nih.gov/articles/PMC6731745
- Sinha et Kapur (2021) — https://journals.sagepub.com/doi/10.3102/00346543211019105
- Prédire-observer-expliquer — https://learningandteaching.anu.edu.au/resources/predict-observe-explain-activity/
- Débogage comme apprentissage — https://arxiv.org/pdf/2503.18080

**Outils**
- TensorFlow Playground — https://github.com/tensorflow/playground ; CNN Explainer — https://arxiv.org/abs/2004.15004 ; Transformer Explainer — https://poloclub.github.io/transformer-explainer ; LLM Visualization — https://github.com/bbycroft/llm-viz ; BertViz — https://github.com/jessevig/bertviz
- Distill — https://distill.pub/2017/momentum ; https://distill.pub/2018/building-blocks
- Netron — https://github.com/lutzroeder/netron ; TensorBoard — https://www.tensorflow.org/tensorboard ; torchview — https://github.com/mert-kurttutan/torchview ; Captum — https://captum.ai ; Ecco — https://github.com/jalammar/ecco
- TransformerLens — https://github.com/TransformerLensOrg/TransformerLens ; circuit-tracer — https://github.com/safety-research/circuit-tracer ; Neuronpedia — https://docs.neuronpedia.org
- micrograd — https://github.com/karpathy/micrograd ; Zero to Hero — https://github.com/karpathy/nn-zero-to-hero
- MLflow — https://mlflow.org/docs/latest/genai ; DVC — https://doc.dvc.org/start ; Great Expectations — https://docs.greatexpectations.io ; Evidently — https://docs.evidentlyai.com/introduction
- Langfuse — https://langfuse.com/docs ; LangSmith — https://docs.langchain.com/langsmith/home ; Phoenix — https://arize.com/docs/phoenix ; Ragas — https://docs.ragas.io ; DeepEval — https://deepeval.com/docs/getting-started ; promptfoo — https://www.promptfoo.dev/docs/intro ; Inspect — https://inspect.aisi.org.uk ; marimo — https://docs.marimo.io
