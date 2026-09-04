from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import time
import os
import httpx
import random
import logging

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    name: str
    pricing_in: float
    pricing_out: float

@dataclass
class Call:
    text: str | None
    error: str | None
    prompt_tokens: int = 0 
    completion_tokens: int = 0
    cost_usd: float = 0
    latency_ms: float = 0
    service_ms: float = 0
    attempts: int = 0

@dataclass
class Stats:
    success_rate: int
    total_cost : float
    total_tokens: float
    p50_latency: float
    p95_latency: float
    p50_service: float
    p95_service: float

class LLMClient:
    def __init__(
            self, 
            model: ModelInfo,
            api_key: str = GEMINI_API_KEY, 
            concurrency: int = 3, 
            max_tentative: int = 3,
            client: httpx.AsyncClient | None = None
    ):
        self.model = model
        self.api_key = api_key
        self._sem = asyncio.Semaphore(concurrency)
        self.max_tentative = max_tentative
        self._http = client
        self._owns_http = client is None

    async def __aenter__(self):
        # ouvrir les ressources, retourner self
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=60,
                headers={"x-goog-api-key": self.api_key}
            )  
        return self
    
    async def __aexit__(self,  exc_type, exc, tb):
        # fermer les ressources
        if self._owns_http:
            await self._http.aclose()

    async def complete(self, prompt: str) -> Call:
        t0_latency = time.perf_counter() 
        async with self._sem:      
            t0_service = time.perf_counter()
            for tentative in range(1, self.max_tentative + 1):
                retry_after = None
                logger.debug("retry %d/%d", tentative, self.max_tentative)
                try:   
                    r = await self._http.post(
                        "https://generativelanguage.googleapis.com/v1beta/interactions",
                        json={"model": self.model.name, "input": f"{prompt}"}
                    )
        
                except (httpx.RequestError) as e:
                    logger.warning("error network : %s", type(e).__name__)
        
                else:
                    status = sort_status(r.status_code)

                    # Pas d'erreur
                    if status == "ok":
                        donnees = r.json()
                        #print(json.dumps(donnees, indent=2))
                        t1_latency = time.perf_counter() - t0_latency
                        t1_service = time.perf_counter() - t0_service
                        return Call(
                            text = extract_text(donnees),
                            error = None,
                            prompt_tokens = donnees["usage"]["total_input_tokens"],
                            completion_tokens = donnees["usage"]["total_output_tokens"],
                            cost_usd = compute_cost(donnees["usage"]["total_input_tokens"], donnees["usage"]["total_output_tokens"], self.model),
                            latency_ms = t1_latency * 1000,
                            service_ms = t1_service * 1000,
                            attempts = tentative
                        )
                    
                    # Erreur def
                    if status == "definitive":
                        return Call(
                            text = None,
                            error = f"Erreur def : {r.status_code}",
                            attempts = tentative
                        )
                    
                    # Erreur temp
                    retry_after = r.headers.get("retry-after")
                    logger.warning("error temp : %d", r.status_code)
        
                # point d'attente unique
                if tentative < self.max_tentative:
                    delay = compute_delay(tentative, retry_after)
                    logger.warning("retry in %.1fs", delay)
                    await asyncio.sleep(delay)

            logger.error("fail after %d attempts", self.max_tentative)
            return Call(
                text = None,
                error = f"fail after {self.max_tentative} attempts",
                attempts = tentative
            )

    async def complete_many(self, prompts: list[str]) -> list[Call]:
        return await asyncio.gather(*[self.complete(p) for p in prompts])


def compute_cost(prompt_tokens: int, completion_tokens: int, model: ModelInfo) -> float:
    cost = (prompt_tokens * model.pricing_in + completion_tokens * model.pricing_out) / 1_000_000
    return cost

def sort_status(status_code: int) -> str:
    match status_code:
        case 200:
            return "ok"
        case 400 | 401 | 403 | 404:
            return "definitive"
        case _:
            return "transitional"

def compute_delay(attempt: int, retry_after: str | None) -> float:
    delay = float(retry_after) if retry_after else random.uniform(1, 1+(2 ** (attempt - 1)))
    return delay

def extract_text(data: dict) -> str:
    values = data["steps"][1]["content"][0]["text"]
    return values

def percentiles(values: list[float]) -> tuple[float, float]:
    n = len(values)
    sorted_values = sorted(values)
    if n == 0:
        return 0.0, 0.0
    p50 = sorted_values[int(0.5 * (n - 1))]
    p95 = sorted_values[int(0.95 * (n - 1))]
    return p50, p95

def stats(calls: list[Call]) -> Stats | None:
    success_rate: int = 0
    total_cost : float = 0
    total_tokens: float = 0

    if calls == []:
        return None
    
    for call in calls:
        if call.error is None:
            success_rate += 1
        total_cost += call.cost_usd
        total_tokens += (call.prompt_tokens + call.completion_tokens)

    success_calls = [c for c in calls if c.error is None]

    p50_latency, p95_latency = percentiles([c.latency_ms for c in success_calls])
    p50_services, p95_services = percentiles([c.service_ms for c in success_calls])

    return Stats(
        success_rate / len(calls), 
        total_cost, 
        total_tokens, 
        p50_latency, 
        p95_latency,
        p50_services, 
        p95_services
    )