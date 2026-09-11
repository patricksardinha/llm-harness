from llm_client import GEMINI_API_KEY, LLMClient, ModelInfo
from retrieval import CORPUS, QUESTIONS, mat, search_reranked
import asyncio

def prompt_builder(documents, question):
    p = f"""
    Réponds à la question en te basant uniquement sur les documents ci-dessous.
    Si l'information n'y figure pas, dis-le.

    Documents :
    {documents}

    Question : {question}"""
    return p

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
    documents = "\n".join(f"[{i}] {doc}" for i, (_, _, doc) in enumerate(top, 1))
    prompt = prompt_builder(documents, question)
    call = await client.complete(prompt)
    return (call.text, [idx for _, idx, _ in top])

async def demo():
    gem_3_1_flash_lite = ModelInfo("gemini-3.1-flash-lite", 0.25, 1.50)
    #q = "Comment les amateurs peuvent-ils se qualifier pour les championnats du monde ?"
    q = "Quelle est la longueur totale d'un Ironman ?"
    async with LLMClient(model=gem_3_1_flash_lite, api_key=GEMINI_API_KEY) as client:
        texte, idx = await answer(q, client)
        print(f"Question: {q}\nResponse: {texte}\nSource: {idx}\n")

if __name__ == "__main__":
    asyncio.run(demo())