from ollama import generate
import asyncio
import time

async def appel_bidon(n: int) -> str:
    await asyncio.sleep(1)        # simule l'attente réseau
    return f"résultat {n}"

async def main():
    debut = time.perf_counter()
    sem = asyncio.Semaphore(3)

    async def borne(n: int) -> str:
        async with sem:                                 
            await asyncio.gather(appel_bidon(n))

    resultats = await asyncio.gather(*[borne(i) for i in range(10)])
    print(f"{time.perf_counter() - debut:.1f}s")

asyncio.run(main())