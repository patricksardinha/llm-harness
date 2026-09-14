from llm_client import GEMINI_API_KEY, LLMClient, ModelInfo
from retrieval import CORPUS, QUESTIONS, mat, search_reranked
import asyncio
import json

CAS = json.load(open("eval_set.json", encoding="utf-8"))


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


def evaluate(cas: dict, texte: str, sources: list[int]) -> dict:
    retrieval = cas["doc_attendu"] in sources
    keywords = all(keyword.lower() in texte.lower() for keyword in cas["mots_cles"])
    not_empty = len(texte.strip()) > 10
    res = {
        "question": cas["question"],
        "texte": texte,
        "sources": sources,
        "retrieval": retrieval,
        "keywords": keywords,
        "not_empty": not_empty
    }
    return res
    

async def answer(question: str, client: LLMClient) -> tuple[str, list[int]]:
    top = search_reranked(question, CORPUS, mat, k=3)
    documents = "\n".join(f"[{i}] {doc}" for i, (_, _, doc) in enumerate(top, 1))
    prompt = prompt_builder(documents, question)
    call = await client.complete(prompt)
    return (call.text, [idx for _, idx, _ in top])


async def get_results(cas, client: LLMClient) -> list[dict]:
    res = []
    for c in cas:
        texte, idx = await answer(c["question"], client)
        res_eval = evaluate(cas=c, texte=texte, sources=idx)
        res.append(res_eval)
    return res

def display_results(results):
    n = len(results)
    for i, c in enumerate(results):
        print(f"[{i}] Question: {c['question']}\nResponse: {c['texte']}\nSource: {c['sources']}\n\n")
    print(f"Retrieval : {sum(1 for r in results if r['retrieval']) / n:.0%}")
    print(f"Keywords  : {sum(1 for r in results if r['keywords']) / n:.0%}")
    print(f"Not empty  : {sum(1 for r in results if r['not_empty']) / n:.0%}")

async def demo():
    gem_3_1_flash_lite = ModelInfo("gemini-3.1-flash-lite", 0.25, 1.50)
    
    async with LLMClient(model=gem_3_1_flash_lite, api_key=GEMINI_API_KEY) as client:
        results = await get_results(CAS, client)
        display_results(results)

        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # load_results = json.load(open("results.json", encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(demo())