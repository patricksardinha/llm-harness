# llm-harness

Harness Python pour appels concurrents à des API de LLM : concurrence bornée,
politique de retry, et mesure systématique.

## Fonctionnalités

- Appels asynchrones (`httpx.AsyncClient`), testés sur l'API Gemini et sur
  Ollama en local
- Limitation de la concurrence par `asyncio.Semaphore`
- Retry avec backoff exponentiel et jitter, distinction entre erreurs
  transitoires et définitives, respect de l'en-tête `Retry-After`
- Aucune exception unitaire ne remonte : N prompts en entrée, N résultats en
  sortie
- Mesure du temps par appel et du temps total

---

## Résultats mesurés

### Concurrence

API Gemini, 10 appels, `Semaphore(3)` :

| Métrique | Valeur |
|---|---|
| Somme des latences individuelles | 263,2 s |
| Temps total | 90,8 s |
| Rapport | 2,9 |

Le rapport de 2,9 pour une limite fixée à 3 confirme que le sémaphore tient
sa borne.

Les latences individuelles s'échelonnent de 9,4 s à 62,4 s. L'écart ne vient
pas du serveur mais de la longueur de la réponse générée : le *prefill*
(traitement du prompt) est parallélisable et quasi constant ici, alors que le
*decode* est séquentiel — une passe complète du modèle par token produit.
« Capitale de l'Australie ? » et « 3 idées de films SF » ont le même prompt en
taille, mais l'un génère trois mots et l'autre une page.

### Contre-mesure : inférence locale

Même code sur Ollama (`llama3.2:1b`, CPU, 7,6 Go de RAM) : **291 s** à
concurrence 3, contre environ 57 s attendus en séquentiel — la concurrence
**dégrade** les performances.

L'inférence locale est CPU-bound : trois requêtes simultanées se partagent le
même processeur au lieu de superposer des attentes réseau. La concurrence
côté client n'ajoute du débit que si le serveur a de la capacité à donner.

### Politique de retry

Mesure du chemin d'échec, `timeout=0.001` pour forcer des `ConnectTimeout`,
10 appels à concurrence 3, 3 tentatives chacun :

| Version | Temps total |
|---|---|
| Attente appliquée après la dernière tentative | 61,4 s |
| Attente conditionnée à `tentative < max_tentatives` | 14,1 s |

Les deux runs diffèrent aussi par la formule du délai, donc le gain n'est pas
entièrement imputable au garde-fou. En attribuant par le calcul (les appels
étant instantanés, le temps mesuré n'est que du sommeil) : la suppression de
l'attente superflue représente environ **55 % du temps d'échec initial**, le
reste venant du raccourcissement des délais.

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

Paramètres modifiables dans `main.py` :

- `nb_call` — nombre de requêtes à envoyer
- `inputs` — liste des prompts
- `asyncio.Semaphore(3)` — nombre maximum d'appels simultanés
- `max_tentatives` — nombre de tentatives par appel

## Structure

```
llm-harness/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`.venv/` et `.env` ne sont pas versionnés.