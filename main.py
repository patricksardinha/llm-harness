import asyncio
import time

from llm_client import GEMINI_API_KEY, LLMClient, ModelInfo, stats

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

    # Model infos: [Name, Pricing in, Pricing out]
    gem_3_1_flash_lite = ModelInfo("gemini-3.1-flash-lite", 0.25, 1.50)
    gem_3_5_flash_lite = ModelInfo("gemini-3.5-flash-lite", 1.50, 9.00)
    gem_3_6_flash = ModelInfo("gemini-3.6-flash", 1.50, 9)

    t0 = time.perf_counter()
    async with LLMClient(model=gem_3_1_flash_lite, api_key=GEMINI_API_KEY) as client:
        t1 = time.perf_counter()
        resultats = await client.complete_many(inputs[4:5])
        t2 = time.perf_counter()     
    t3 = time.perf_counter()     
    
    print(f"ouverture {t1-t0:.1f} | appels {t2-t1:.1f} | fermeture {t3-t2:.1f}")

    print(f"Total time : {time.perf_counter() - debut:.1f}s")
    print(resultats)

    statistics = stats(resultats)
    print(statistics)

if __name__ == "__main__":
    asyncio.run(main())