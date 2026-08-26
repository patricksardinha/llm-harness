import os
from dotenv import load_dotenv
import asyncio
import time
import httpx

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

async def call_llm(n: int, client: httpx.AsyncClient, inputs: list[str]) -> str:
    debut_simple_call = time.perf_counter()    
    r = await client.post(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        json={"model": "gemini-3.6-flash", "input": f"{inputs[n]}"}
    )
    print(f"{time.perf_counter() - debut_simple_call:.1f}s")

    if r.status_code != 200:
        print(r.status_code, r.text)
        return f"ERREUR {r.status_code}"
    
    donnees = r.json()
    texte = donnees["steps"][1]["content"][0]["text"]
    return texte

async def main():
    nb_call = 10
    inputs = [
        "Explique la photosynthèse.",
        "Recette de pâte à crêpes ?",
        "Différence étoile vs planète ?",
        "Résumé du Petit Prince.",
        "Traduire 'où sont les toilettes' en espagnol.",
        "Pourquoi le ciel est bleu ?",
        "3 idées de films SF.",
        "Capitale de l'Australie ?",
        "Corrige : 'Je sui allé au marché'.",
        "Une blague courte sur les chats."
    ]
    debut = time.perf_counter()
    sem = asyncio.Semaphore(3)

    async def borne(n: int, client: httpx.AsyncClient) -> str:                       
        async with sem:      
            return await call_llm(n, client, inputs)

    async with httpx.AsyncClient(
        timeout=300,
        headers={"x-goog-api-key": GEMINI_API_KEY}
    ) as client:
        resultats = await asyncio.gather(*[borne(i, client) for i in range(nb_call)])
        print(resultats)

    print(f"{time.perf_counter() - debut:.1f}s")

asyncio.run(main())