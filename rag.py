from llm_client import GEMINI_API_KEY, LLMClient, ModelInfo
from retrieval import CORPUS, QUESTIONS, mat, search_reranked
import asyncio

def judge_prompt(documents, question, reponse):
    instruction = f"""
    Tu évalues si une réponse est entièrement appuyée sur les documents fournis.

    Documents : {documents}
    Question : {question}
    Réponse : {reponse}

    Réponds "oui" si chaque affirmation de la réponse figure dans les documents,
    "non" si la réponse contient une information absente des documents.
    Format : {{"verdict": "oui"|"non", "raison": "..."}} """
    return instruction
    

async def answer(question: str, client: LLMClient) -> tuple[str, list[int]]:
    top = search_reranked(question, CORPUS, mat, k=3)
    return

async def demo():
    gem_3_1_flash_lite = ModelInfo("gemini-3.1-flash-lite", 0.25, 1.50)
    async with LLMClient(model=gem_3_1_flash_lite, api_key=GEMINI_API_KEY) as client:
        texte, idx = await answer(q, client)
        print()

if __name__ == "__main__":
    q = "Comment les amateurs peuvent-ils se qualifier pour les championnats du monde ?"
    asyncio.run(demo())