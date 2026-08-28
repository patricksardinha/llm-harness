import os
from dotenv import load_dotenv
import asyncio
import time
import httpx
import random

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

class LLMClient:
    def __init__(self, model: str, api_key: str = GEMINI_API_KEY, concurrency: int = 3, max_tentative: int = 3):
        self.model = model
        self.api_key = api_key
        self._sem = asyncio.Semaphore(concurrency)
        self.max_tentative = max_tentative

    async def __aenter__(self):
        # ouvrir les ressources, retourner self
        self._http = httpx.AsyncClient(
            timeout=60,
            headers={"x-goog-api-key": self.api_key}
        )  
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        # fermer les ressources
        await self._http.aclose()

    async def complete(self, prompt: str) -> str:
        async with self._sem:      
            for tentative in range(1, self.max_tentative + 1):
                retry_after = None
                print(f"Tentatives : {tentative}/{self.max_tentative}")
                try:   
                    debut_simple_call = time.perf_counter() 
                    r = await self._http.post(
                        "https://generativelanguage.googleapis.com/v1beta/interactions",
                        json={"model": self.model, "input": f"{prompt}"}
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
                if tentative < self.max_tentative:
                    delai = float(retry_after) if retry_after else random.uniform(1, 1+(2 ** (tentative - 1)))
                    print(f"Wainting {delai}[s]")
                    await asyncio.sleep(delai)
        
            return f"Echec après {self.max_tentative} tentatives"

    async def complete_many(self, prompts: list[str]) -> list[str]:
        return await asyncio.gather(*[self.complete(p) for p in prompts])
    

async def main():
    debut = time.perf_counter()        
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

    async with LLMClient("gemini-3.6-flash", api_key=GEMINI_API_KEY) as client:
        resultats = await client.complete_many(inputs)

    print(f"{time.perf_counter() - debut:.1f}s")
    print(resultats)

asyncio.run(main())