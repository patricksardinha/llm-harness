import pytest
import httpx
from llm_client import ModelInfo, LLMClient, compute_cost, sort_status, compute_delay, extract_text

RESPONSE_OK = {
    "steps": [{"type": "thought"}, {"content": [{"text": "Hello"}]}],
    "usage": {"total_input_tokens": 10, "total_output_tokens": 20},
}

def test_compute_cost():
    model = ModelInfo("fake", 1.0, 2.0)
    result = compute_cost(1000, 2000, model)      
    assert result == 0.005

@pytest.mark.parametrize("code, attendu", [
    (200, "ok"),
    (400, "definitive"),
    (401, "definitive"),
    (404, "definitive"),
    (429, "transitional"),
    (500, "transitional"),
    (503, "transitional"),
])
def test_sort_status(code, attendu):
    result = sort_status(code)
    assert result == attendu

def test_compute_delay():
    result = compute_delay(attempt=1, retry_after=None)
    assert 1 <= result <= 2

def test_compute_delay_bornes(monkeypatch):
    appels = []
    def faux_uniform(a, b):
        appels.append((a, b))
        return 3.0
    monkeypatch.setattr("llm_client.random.uniform", faux_uniform)

    result = compute_delay(attempt=3, retry_after=None)

    assert result == 3.0
    assert appels == [(1, 5)]

def test_compute_delay_with_retry():
    result = compute_delay(attempt=3, retry_after="7")
    assert result == 7.0

def test_extract_text():
    donnees = {"steps": [{"type": "thought"}, {"content": [{"text": "Hello"}]}]}
    assert extract_text(donnees) == "Hello"

"""
Tests sequences
"""

def handler_code(code: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code)
    return handler

def handler_sequence(reponses: list[httpx.Response]):
    compteur = {"n": 0}
    def handler(request: httpx.Request):
        r = reponses[compteur["n"]]
        compteur["n"] += 1
        return r
    return handler

@pytest.mark.asyncio
async def test_400_not_trigger_retry():
    transport = httpx.MockTransport(handler_code(code=400))
    async with LLMClient(
        ModelInfo("fake", 1.0, 2.0),
        client=httpx.AsyncClient(transport=transport),
    ) as client:
        call = await client.complete("Fake prompt")

    assert call.error is not None
    assert call.attempts == 1


@pytest.mark.asyncio
async def test_3_500_trigger_max_tentative(monkeypatch):
    transport = httpx.MockTransport(handler_code(500))
    sleep_time = []
    async def faux_asyncio_sleep(a):
        sleep_time.append(a)
        return 0
    monkeypatch.setattr("llm_client.asyncio.sleep", faux_asyncio_sleep)

    async with LLMClient(
        ModelInfo("fake", 1.0, 2.0),
        client=httpx.AsyncClient(transport=transport),
    ) as client:
        call = await client.complete("Fake prompt")

    assert len(sleep_time) == 2
    assert call.text is None
    assert call.error is not None
    assert call.attempts == 3

@pytest.mark.asyncio
async def test_429_then_200_successed(monkeypatch):
    transport = httpx.MockTransport(handler_sequence([httpx.Response(429), httpx.Response(200, json=RESPONSE_OK)]))
    sleep_time = []
    async def faux_asyncio_sleep(a):
        sleep_time.append(a)
        return 0
    monkeypatch.setattr("llm_client.asyncio.sleep", faux_asyncio_sleep)

    async with LLMClient(
        ModelInfo("fake", 1.0, 2.0),
        client=httpx.AsyncClient(transport=transport),
    ) as client:
        call = await client.complete("Fake prompt")

    assert call.text == "Hello"
    assert call.error is None
    assert call.attempts == 2
    assert call.cost_usd == ((1.0 * 10) + (2 * 20)) / 1_000_000

@pytest.mark.asyncio
async def test_sequence_200_400_200(monkeypatch):
    transport = httpx.MockTransport(handler_sequence([httpx.Response(200, json=RESPONSE_OK), httpx.Response(400), httpx.Response(200, json=RESPONSE_OK)]))
    sleep_time = []
    results = []
    async def faux_asyncio_sleep(a):
        sleep_time.append(a)
        return 0
    monkeypatch.setattr("llm_client.asyncio.sleep", faux_asyncio_sleep)

    async with LLMClient(
        ModelInfo("fake", 1.0, 2.0),
        client=httpx.AsyncClient(transport=transport),
    ) as client:
        call = await client.complete("Fake prompt")

    assert len(results) == 3
