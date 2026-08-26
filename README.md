# llm-harness

Harness Python pour appels concurrents à des API de LLM.

## Fonctionnalités

- Appels asynchrones à l'API Gemini / Ollama via `httpx.AsyncClient`
- Limitation de la concurrence avec `asyncio.Semaphore`
- Mesure du temps d'exécution (par appel et total)

## Résultats mesurés

API Gemini, 10 appels, `Semaphore(3)` :

| Métrique | Valeur |
|---|---|
| Somme des latences individuelles | 263,2 s |
| Temps total (concurrence 3) | 90,8 s |
| Rapport | 2,9 |

Latences individuelles : 9,4 s à 62,4 s selon la longueur de la réponse
générée - le prefill est quasi constant, le decode est séquentiel (une passe
du modèle par token produit).

**Contre-mesure sur modèle local (Ollama, llama3.2:1b)** : la concurrence
dégrade les performances (291 s à concurrence 3, contre ~57 s attendus en
séquentiel). L'inférence locale est CPU-bound : trois requêtes se partagent
le même processeur. La concurrence côté client n'ajoute du débit que si le
serveur a de la capacité à donner. 

## Prérequis

- Python 3.11+
- Une clé API Gemini (free tier)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Configuration

Créer un fichier `.env` à la racine :

```
GEMINI_API_KEY=votre_clé_ici
```

## Utilisation

```bash
python main.py
```

Paramètres modifiables dans `main.py` :

- `nb_call` : nombre de requêtes à envoyer
- `inputs` : liste des prompts
- `asyncio.Semaphore(3)` : nombre maximum d'appels simultanés

## Structure du projet

```
llm-harness/
├── .venv  
├── .env   
├── .gitignore   
├── main.py      
├── requirements.txt   
└── README.md
```
