import os
from dotenv import load_dotenv
import asyncio
import time
import httpx
import random

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

async def call_llm_with_retry(n: int, client: httpx.AsyncClient, inputs: list[str], max_tentatives: int = 3) -> str:
    for tentative in range(1, max_tentatives + 1):
        retry_after = None
        print(f"Tentatives : {tentative}/{max_tentatives}")
        try:   
            debut_simple_call = time.perf_counter() 
            r = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                json={"model": "gemini-3.6-flash", "input": f"{inputs[n]}"}
            )
            print(f"Temps une requête: {time.perf_counter() - debut_simple_call:.1f}s")

        except (httpx.RequestError) as e:
            print(f"Erreur réseau : {type(e).__name__}")

        else:
            # Pas d'erreur
            if r.status_code == 200:
                donnees = r.json()
                return donnees["steps"][1]["content"][0]["text"]
            
            # Erreur def
            if (r.status_code in (400, 401, 403, 404)):
                return f"Erreur def : {r.status_code}"
            
            # Erreur temp
            retry_after = r.headers.get("retry-after")
            print(f"Erreur temp : {r.status_code}")

        # point d'attente unique
        if tentative < max_tentatives:
            delai = float(retry_after) if retry_after else random.uniform(1, 1+(2 ** (tentative - 1)))
            print(f"Wainting {delai}[s]")
            await asyncio.sleep(delai)

    return f"Echec après {max_tentatives} tentatives"
    

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
            return await call_llm_with_retry(n, client, inputs)

    async with httpx.AsyncClient(
        timeout=0.001,
        headers={"x-goog-api-key": GEMINI_API_KEY}
    ) as client:
        resultats = await asyncio.gather(*[borne(i, client) for i in range(nb_call)])
        print(resultats)

    print(f"{time.perf_counter() - debut:.1f}s")

asyncio.run(main())