# llm-harness

Client Python pour appels concurrents à des API de LLM : concurrence bornée,
politique de retry, comptabilité des tokens et du coût, et mesure systématique.

## Fonctionnalités

- Client asynchrone (`httpx.AsyncClient`), testé sur l'API Gemini et sur Ollama
  en local
- Limitation de la concurrence par `asyncio.Semaphore` partagé
- Retry avec backoff exponentiel et jitter, distinction entre erreurs
  transitoires et définitives, respect de l'en-tête `Retry-After`
- Aucune exception unitaire ne remonte : N prompts en entrée, N résultats en
  sortie
- Comptabilité par appel : tokens d'entrée et de sortie, coût en dollars,
  latence perçue, temps de service, nombre de tentatives
- Statistiques agrégées : taux de succès, coût total, percentiles
- Client HTTP injectable, pour tester sans réseau

---

## Résultats mesurés

### Concurrence

API Gemini, 10 appels, `Semaphore(3)` :

| Métrique | Valeur |
|---|---|
| Somme des latences individuelles | 263,2 s |
| Temps total | 90,8 s |
| Rapport | 2,9 |

Le rapport de 2,9 pour une limite fixée à 3 confirme que le sémaphore tient sa
borne.

### Contre-mesure : inférence locale

Même code sur Ollama (`llama3.2:1b`, CPU, 7,6 Go de RAM) : **291 s** à
concurrence 3, contre environ 57 s attendus en séquentiel — la concurrence
**dégrade** les performances.

L'inférence locale est CPU-bound : trois requêtes simultanées se partagent le
même processeur au lieu de superposer des attentes réseau. La concurrence côté
client n'ajoute du débit que si le serveur a de la capacité à donner.

### Politique de retry

Chemin d'échec mesuré avec `timeout=0.001` pour forcer des `ConnectTimeout`,
10 appels à concurrence 3, 3 tentatives chacun :

| Version | Temps total |
|---|---|
| Attente appliquée après la dernière tentative | 61,4 s |
| Attente conditionnée à `tentative < max_tentatives` | 14,1 s |

Les deux runs diffèrent aussi par la formule du délai, donc le gain n'est pas
entièrement imputable au garde-fou. En attribuant par le calcul (les appels
étant instantanés, le temps mesuré n'est que du sommeil), la suppression de
l'attente superflue représente environ **55 % du temps d'échec initial**.

### Coût

10 appels sur `gemini-3.1-flash-lite`, 89 tokens en entrée et 4 389 en sortie :

| Métrique | Valeur |
|---|---|
| Coût total | 0,0066 $ |
| Coût projeté pour 1 000 appels | **0,66 $** |
| Taux de succès | 100 % |

### Latence perçue et temps de service

Deux mesures distinctes, chronométrées de part et d'autre de l'acquisition du
sémaphore :

| Métrique | p50 | p95 |
|---|---|---|
| Latence perçue (file d'attente incluse) | 13,8 s | 16,8 s |
| Temps de service (appel seul) | 4,9 s | 8,0 s |

L'écart est l'attente en file. Le dernier prompt soumis a patienté près de
14 secondes avant même de partir. Les confondre empêche tout diagnostic : un p95
dégradé par une file trop longue et un p95 dégradé par un provider lent se
corrigent de manières opposées.

**Conséquence pratique.** Les deux cas se distinguent par l'écart entre les deux
mesures, pas par la latence perçue seule :

- `latency_ms` élevé, `service_ms` normal → le goulot est **chez moi**. Les
  requêtes attendent leur jeton alors que le provider répond vite. Correction :
  **augmenter** la concurrence (ou paralléliser davantage). C'est le cas mesuré
  ici : 13,8 s perçus pour 4,9 s de service.
- `latency_ms` et `service_ms` élevés tous les deux → le goulot est **chez le
  provider**. Correction : **réduire** la concurrence, sous peine de déclencher
  des 429 et d'aggraver la situation ; puis raccourcir les sorties, router vers
  un modèle plus rapide, ou mettre en cache.

Appliquer la mauvaise correction empire le problème dans les deux sens :
augmenter la concurrence face à un provider saturé provoque des rejets, et la
réduire face à une file trop longue allonge encore l'attente.

Percentiles calculés sur les seuls succès, méthode du rang le plus proche par
troncature. **Sur n = 10, le p95 vaut la 9ᵉ valeur triée : indicatif seulement.**

### Coût fixe par appel

Débit de génération selon la longueur de la réponse :

| Prompt | Tokens de sortie | Service | Débit |
|---|---|---|---|
| Résumé du Petit Prince | 821 | 7,96 s | 103 tok/s |
| Explique la photosynthèse | 669 | 6,25 s | 107 tok/s |
| Une blague sur les chats | 36 | 2,67 s | 13 tok/s |
| Capitale de l'Australie | 12 | 2,31 s | 5 tok/s |

Le débit s'effondre sur les réponses courtes : il existe un coût fixe payé quel
que soit le volume (aller-retour réseau, mise en file côté provider, prefill,
premier token). Le modèle qui décrit ces mesures est
`service ≈ 2 200 ms + tokens × 7 ms` — vérifié à 11 ms près sur le Petit Prince
(7 947 prédit contre 7 958 mesuré).

Conséquence pratique : sur des réponses courtes, optimiser la génération ne sert
à rien, tout est dans l'overhead. Il faut batcher ou mettre en cache.

### Embeddings : la limite de la recherche vectorielle

Similarité cosinus calculée à la main (produit scalaire divisé par le produit
des normes), sur trois configurations de modèle et de langue :

| Paire | MiniLM-L6 (EN) | Multilingue (EN) | Multilingue (FR) |
|---|---|---|---|
| chat / félin | 0,698 | 0,556 | 0,208 |
| chat / voiture | 0,463 | 0,351 | 0,412 |
| **BZ-4471-A / BZ-4471-B** | **0,981** | **0,983** | **0,983** |

Deux **références produit distinctes** obtiennent un score de quasi-identité,
stable sur les trois configurations. Une référence n'est pas du langage : elle
est découpée en fragments de sous-mots qui ne portent aucun sens, et deux codes
voisins deviennent indiscernables dans l'espace vectoriel.

Le modèle est donc **très sûr là où il a tort** (0,98 pour deux pièces
différentes) et **hésitant là où il a raison** (0,21 pour deux mots liés). Sur
un corpus industriel, l'information discriminante — références, codes pièce,
numéros de norme, acronymes internes — est précisément celle que les embeddings
ignorent. C'est la justification empirique de la recherche hybride
(BM25 + dense), et l'explication probable de l'échec d'un précédent prototype
RAG.

Deux effets secondaires mesurés, également utiles :

- **Le choix du modèle est un levier de qualité.** `all-MiniLM-L6-v2` est
  anglophone ; les scores s'effondrent sur des entrées françaises.
- **Un modèle peut se tromper sans le moindre signal.** En français, le modèle
  multilingue place « chat » plus près de « voiture » que de « félin ». Cause
  probable : ces modèles sont entraînés sur des *phrases*, et des mots isolés
  sortent de leur distribution d'entraînement.

### Retrieval : mesure et reranking

Corpus de 20 paragraphes, jeu d'évaluation de 20 questions écrites à la main
(chaque question a un document attendu). Index en mémoire, similarité cosinus
calculée à la main, sans base vectorielle.

| | recall@1 | recall@3 | Temps (20 questions) |
|---|---|---|---|
| Bi-encodeur seul | 70 % | 100 % | 0,6 s |
| + cross-encodeur (10 candidats) | **85 %** | 100 % | 7,1 s |

**Le diagnostic tient dans l'écart entre les deux colonnes.** Un recall@3 de
100 % pour un recall@1 de 70 % signifie que le bon document est *toujours*
récupéré, mais mal classé une fois sur trois. Ce n'est pas un problème de
récupération — chunking et modèle d'embedding font leur travail — c'est un
problème de **classement**. C'est précisément ce qu'un reranker corrige, et
c'est pourquoi il fallait mesurer avant de choisir quoi améliorer.

**Bi-encodeur contre cross-encodeur.** Le premier encode requête et documents
séparément, puis compare les vecteurs : les documents sont encodés une fois pour
toutes, la recherche est quasi instantanée. Mais au moment de comprimer un
document en 384 nombres, il ignore quelle sera la question. Le second traite la
paire ensemble en une passe, voit les mots de la question face à ceux du
document — bien plus précis, mais impossible à pré-calculer. D'où l'architecture
en deux temps : le bi-encodeur ramène 10 candidats, le cross-encodeur les
reclasse.

**Le coût mesuré : ×12** (0,6 s → 7,1 s pour 20 questions, soit ~355 ms par
requête). C'est ce facteur qui interdit de reranker tout un corpus, et qui
justifie l'étage de présélection.

**Une régression cachée par la moyenne.** Le reranker corrige 4 échecs mais en
introduit 1 nouveau (« quel type de vélo pour longue distance »), soit +3 net.
Une métrique agrégée aurait affiché 70 % → 70 % si les régressions avaient
compensé les corrections, en masquant un changement de comportement complet.
La comparaison doit se faire question par question, pas moyenne contre moyenne.

**Le phénomène de l'aimant.** Un chunk générique capte les requêtes sans terme
discriminant : le paragraphe de généralités concentrait 3 des 6 échecs du
bi-encodeur. Après reranking le phénomène ne disparaît pas, il se déplace — le
paragraphe « Ironman XXL » capte les 3 échecs restants.

**Les échecs résiduels sont des quasi-doublons** : half-Ironman contre Ironman,
distance olympique contre XXL. Un mot discriminant pèse trop peu dans un vecteur
qui résume trois phrases — même constat que sur les références produit
ci-dessus. C'est le cas d'usage de BM25 et de la recherche hybride.

Deux réserves méthodologiques : sur 20 documents, retenir les 3 premiers revient
à garder 15 % du corpus, ce qui rend le recall@3 de 100 % moins impressionnant
qu'il n'y paraît ; et les temps sont mesurés après un tour de chauffe, sur une
seule exécution.

### Évaluation : pipeline RAG et juge calibré

Pipeline complet : `search_reranked` → injection des 3 passages dans le prompt →
génération. Jeu d'évaluation de 20 questions versionné dans `eval_set.json`,
avec pour chacune le document attendu et les mots-clés obligatoires.

#### Assertions déterministes

| Métrique | Valeur |
|---|---|
| Retrieval (le bon document est dans le contexte) | 100 % |
| Mots-clés obligatoires présents | 100 % |
| Réponse non vide | 100 % |

**Ces 100 % ne sont pas un bon résultat, c'est un jeu d'éval saturé.** Les
mots-clés ont été ajustés après lecture des réponses pour corriger trois faux
négatifs (une réponse correcte rejetée parce qu'elle employait un synonyme, ou
parce que le jeu exigeait des faits que la question ne demandait pas). Le test
a donc convergé vers la sortie observée et ne peut plus détecter de régression
fine. Il garde sa valeur de test de fumée, pas de mesure de qualité.

Limite structurelle des assertions par mots-clés : faux négatifs sur les
formulations alternatives, faux positifs sur les sous-chaînes fortuites
(`"40"` est contenu dans `"42,195"`). C'est la raison d'être du juge.

#### Calibration du juge (LLM-as-judge, faithfulness)

Critère binaire : chaque affirmation de la réponse est-elle appuyée sur les
documents **effectivement fournis** ? 20 réponses annotées à la main, à
l'aveugle, avant de voir les verdicts du juge. Quatre cas négatifs injectés
volontairement.

| Version | Accord | Détectées | Manquées | Fausses alertes |
|---|---|---|---|---|
| v0 (avant correctif) | **50 %** | — | — | — |
| v1 | 95 % | 3 | 1 | 0 |
| v2 | 90 % | 2 | 2 | 0 |

**La première calibration a trouvé un bug, pas un mauvais juge.** À 50 %
d'accord, neuf désaccords sur dix portaient la même raison : *« la réponse cite
le document [1], or les documents fournis sont [6], [18], [0] »*. Le générateur
numérotait les passages par position (`[1] [2] [3]`) et le juge les
renumérotait par indice dans le corpus. Les deux ne voyaient pas les mêmes
étiquettes. Correctif : la chaîne de documents envoyée au modèle est désormais
**conservée**, pas reconstruite — toute reconstruction peut diverger de
l'original.

**Un accord peut être fortuit.** En v1, le juge et l'annotation humaine étaient
d'accord sur un cas — mais pour des raisons différentes : l'humain avait relevé
une inférence causale, le juge avait relevé que la réponse écrivait « course à
pied » là où le document disait « course ». D'où la règle : lire les raisons,
pas seulement les verdicts.

**La baisse de 95 % à 90 % est une mesure plus honnête, pas une régression.**
La v2 a supprimé le pinaillage lexical (« les reformulations et synonymes sont
acceptables »), ce qui a fait disparaître l'accord fortuit. Le diagnostic
devient net : ce juge détecte les **ajouts explicites** (un nom propre inventé,
un calcul affiché) et rate ce qui relève du raisonnement — une inférence
implicite, ou la suppression d'une condition (une réponse transformant
« privilégié sur les formats longs *sans drafting* » en règle absolue est
passée dans les deux versions).

Les deux types d'erreur n'ont pas le même coût, d'où la matrice plutôt qu'un
taux unique : une hallucination manquée part en production, une fausse alerte
fait seulement perdre du temps.

#### Effet de bord : le retry a sauvé le run

Quatre erreurs 500 pendant la campagne de jugement, quatre reprises après le
délai indiqué par l'en-tête `Retry-After` du provider. Aucun verdict perdu,
aucun run à recommencer.

#### Prochaines étapes identifiées

Le jeu d'évaluation est trop facile pour rester discriminant : il faudrait des
questions sans réponse dans le corpus (test anti-hallucination, le bon
comportement étant le refus), des questions multi-documents, et des questions
piégeuses sur les quasi-doublons. Côté juge, une version demandant d'énumérer
les affirmations une par une avant de trancher forcerait le raisonnement plutôt
que la comparaison lexicale.

### Tests

19 tests, exécutés en **0,26 s**, sans accès réseau et sans clé d'API valide.
Couverture de la bibliothèque : **88 %**.

```bash
pytest -v
pytest --cov=llm_client
```

Le principe retenu : on teste sa propre logique, pas le modèle. La qualité des
réponses relève de l'évaluation, pas des tests unitaires.

**Fonctions pures** — calcul du coût, classification des codes de statut
(7 cas paramétrés), calcul du backoff et priorité de `Retry-After`, extraction
du texte de la réponse, percentiles sur liste vide.

**Logique de retry**, via `httpx.MockTransport` :

| Test | Vérifie |
|---|---|
| 400 | Une seule tentative — aucun retry sur erreur définitive |
| 500 × 3 | Trois tentatives, **deux** attentes seulement |
| 429 puis 200 | Succès en deux tentatives, coût correct |
| `complete_many` avec un 400 | 3 prompts → 3 résultats, exactement une erreur |
| `ConnectError` levée par le transport | Trois tentatives, échec propre par le chemin de l'exception |

Le test des trois 500 vérifie automatiquement le garde-fou mesuré plus haut :
l'attente n'est appliquée qu'entre deux tentatives, jamais après la dernière.

**Effet de bord structurel.** Écrire ces tests a révélé un défaut de découpage :
`complete` mélangeait calcul et entrée-sortie. Quatre fonctions pures en ont été
extraites, et le client HTTP est devenu injectable. Le code s'est réorganisé en
noyau fonctionnel testable et couche impérative mince — la structure a suivi la
testabilité, pas l'inverse.

### Limites connues de la mesure

- Les tokens comptés sont ceux **observés**, pas ceux **facturés** : un échec
  après plusieurs tentatives a pu consommer côté provider sans apparaître ici.
- `total_thought_tokens` et `total_cached_tokens` ne sont pas encore intégrés au
  calcul de coût. Un prompt mesuré à 15 tok/s sort du modèle ci-dessus, ce qui
  suggère du raisonnement interne non comptabilisé.

---

## Prérequis

- Python 3.11+
- Une clé d'API Gemini (offre gratuite suffisante)

## Installation

```bash
python -m venv .venv

.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Configuration

Copier `.env.example` vers `.env` et renseigner la clé :

```
GEMINI_API_KEY=votre_clé_ici
```

## Utilisation

```bash
python main.py
```

Paramètres du client :

- `model` — un `ModelInfo` (nom, prix d'entrée et de sortie par million de
  tokens)
- `concurrency` — nombre maximum d'appels simultanés
- `max_tentative` — nombre de tentatives par appel
- `api_key` — injectable, avec la variable d'environnement comme valeur par
  défaut

## Structure

```
llm-harness/
├── llm_client.py         # bibliothèque : dataclasses, LLMClient, fonctions pures
├── retrieval.py          # index en mémoire, cosinus, bi-encodeur + reranking
├── rag.py                # pipeline RAG, assertions, juge, calibration
├── main.py               # script de démonstration
├── test_llm_client.py    # suite de tests
├── eval_set.json         # jeu d'évaluation (versionné)
├── labels_humains.json   # annotations manuelles de référence (versionné)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`.venv/`, `.env`, `results.json` et `*_judged.json` ne sont pas versionnés :
ce sont des sorties d'exécution. Le jeu d'évaluation et les annotations
humaines, eux, sont des données sources.

`.venv/` et `.env` ne sont pas versionnés.