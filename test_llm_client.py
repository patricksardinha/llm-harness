import pytest
import httpx
from main import ModelInfo, LLMClient, compute_cost, sort_status, compute_delay, extract_text

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
    monkeypatch.setattr("main.random.uniform", faux_uniform)

    result = compute_delay(attempt=3, retry_after=None)

    assert result == 3.0
    assert appels == [(1, 5)]

def test_compute_delay_with_retry():
    result = compute_delay(attempt=3, retry_after="7")
    assert result == 7.0

def test_extract_text():
    donnees = {"steps": [{"type": "thought"}, {"content": [{"text": "bonjour"}]}]}
    assert extract_text(donnees) == "bonjour"

def handler(req):
    return httpx.Response(400)

@pytest.mark.asyncio
async def test_400_without_retry():
    transport = httpx.MockTransport(handler)
    async with LLMClient(
        ModelInfo("fake", 1.0, 2.0),
        client=httpx.AsyncClient(transport=transport),
    ) as client:
        call = await client.complete("peu importe")

    assert call.error is not None
    assert call.attempts == 1