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
├── main.py               # script de démonstration
├── test_llm_client.py    # suite de tests
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`.venv/` et `.env` ne sont pas versionnés.