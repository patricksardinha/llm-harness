from llm_client import GEMINI_API_KEY, LLMClient, ModelInfo
from retrieval import CORPUS, QUESTIONS, mat, search_reranked
import asyncio
import json

JUDGE_PROMPT_VERSION = "v2"
CAS = json.load(open("eval_set.json", encoding="utf-8"))
human_jugement = json.load(open("labels_humains.json", encoding="utf-8")) 

def judge_prompt(documents, question, reponse):
    instruction = f"""
    Tu évalues si une réponse est entièrement appuyée sur les documents fournis.

    Documents : {documents}
    Question : {question}
    Réponse : {reponse}

    Réponds "oui" si chaque affirmation de la réponse figure dans les documents,
    "non" si la réponse contient une information absente des documents.
    Format : {{"verdict": "oui"|"non", "raison": "..."}} 
    
    Considère comme non fidèle toute information déduite, calculée ou inférée 
    qui ne figure pas littéralement dans les documents, même si le calcul est correct.
    
    Les reformulations et synonymes sont acceptables; 
    la suppression d'une condition ou d'une restriction ne l'est pas."""
    return instruction


async def juge(question: str, documents: str, reponse: str, client: LLMClient) -> dict:
    call = await client.complete(judge_prompt(documents, question, reponse))
    jugement = call.text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(jugement)
    except json.JSONDecodeError:
        return {"verdict": None, "raison": "parsing failed", "brut": call.text}

    judge_response = {
        "verdict": {"oui": True, "non": False}.get(data["verdict"]),
        "raison": data.get("raison"),
        "brut": None
    }
    return judge_response


def prompt_builder(documents, question):
    p = f"""
    Réponds à la question en te basant uniquement sur les documents ci-dessous.
    Si l'information n'y figure pas, dis-le.

    Documents :
    {documents}

    Question : {question}"""
    return p


def evaluate(cas: dict, texte: str, sources: list[int], documents: str) -> dict:
    retrieval = cas["doc_attendu"] in sources
    keywords = all(keyword.lower() in texte.lower() for keyword in cas["mots_cles"])
    not_empty = len(texte.strip()) > 10
    res = {
        "question": cas["question"],
        "texte": texte,
        "sources": sources,
        "documents": documents,
        "retrieval": retrieval,
        "keywords": keywords,
        "not_empty": not_empty
    }
    return res
    

async def answer(question: str, client: LLMClient) -> tuple[str, list[int], str]:
    top = search_reranked(question, CORPUS, mat, k=3)
    documents = "\n".join(f"[{i}] {doc}" for i, (_, _, doc) in enumerate(top, 1))
    prompt = prompt_builder(documents, question)
    call = await client.complete(prompt)
    return (call.text, [idx for _, idx, _ in top], documents)


async def get_results(cas, client: LLMClient) -> list[dict]:
    res = []
    for c in cas:
        texte, idx, documents = await answer(c["question"], client)
        res_eval = evaluate(cas=c, texte=texte, sources=idx, documents=documents)
        res.append(res_eval)
    return res

def display_results(results):
    n = len(results)
    for i, c in enumerate(results):
        print(f"[{i}] Question: {c['question']}\nResponse: {c['texte']}\nSource: {c['sources']}\n\n")
    print(f"Retrieval : {sum(1 for r in results if r['retrieval']) / n:.0%}")
    print(f"Keywords  : {sum(1 for r in results if r['keywords']) / n:.0%}")
    print(f"Not empty  : {sum(1 for r in results if r['not_empty']) / n:.0%}")

def display_jugement(question: str, response: str, jugement: dict, i: int): 
    print(f"\nJugement:")
    print(f"Question: {question}")
    print(f"Response: {response}")
    print(f"Verdict juge vs human: {jugement['verdict']} | {human_jugement[i]['info']}")
    print(f"Raison: {jugement['raison']}")

def display_comparaison(comp: dict, total_size: int):
    print(f"\nComparaison (total: {total_size}):")
    print(f"{comp['detected_hallu']} hallucination detected, " 
          f"{comp['missed_hallu']} hallucination missed, "
          f"{comp['false_alert']} false alert, "
          f"{comp['agree']} agreed, "
          f"{comp['failed_judgement']} failed judgements"
          f"\n{((comp['detected_hallu'] + comp['agree']) / total_size):.0%} overall agreement rate")

def compare_judgements(judgements: list[dict], labels: list[dict]) -> dict:
    OUTCOMES = {
        # (judge, humain)
        (False, False): "detected_hallu",   
        (True,  False): "missed_hallu",
        (False, True):  "false_alert",      
        (True,  True):  "agree",            
    }

    comp = dict.fromkeys(OUTCOMES.values(), 0)
    comp["failed_judgement"] = 0

    for j, h in zip(judgements, labels, strict=True):
        if j["verdict"] is None:
            comp["failed_judgement"] += 1
        else: 
            comp[OUTCOMES[bool(j["verdict"]), bool(h["info"])]] += 1
    return comp


async def demo():
    testing_mode = True
    gem_3_1_flash_lite = ModelInfo("gemini-3.1-flash-lite", 0.25, 1.50)
    
    async with LLMClient(model=gem_3_1_flash_lite, api_key=GEMINI_API_KEY) as client:
        if testing_mode:
            loaded_results = json.load(open("results.json", encoding="utf-8"))   
            reuse_judgements = True
            save_judgements = True
            if reuse_judgements:
                save_judgements = False
                judgements = json.load(open(f"{JUDGE_PROMPT_VERSION}_judged.json", encoding="utf-8"))
            else:
                judgements = []
                for i, res in enumerate(loaded_results):
                    quest = res["question"]
                    docs = res["documents"]
                    resp = res["texte"]
                    jugement = await juge(question=quest, documents=docs, reponse=resp, client=client)
                    judgements.append(jugement)
                    display_jugement(quest, resp, jugement, i)

            if save_judgements:
                with open(f"{JUDGE_PROMPT_VERSION}_judged.json", "w", encoding="utf-8") as f:
                    json.dump(judgements, f, ensure_ascii=False, indent=2)

            comp = compare_judgements(judgements, human_jugement)
            display_comparaison(comp, len(judgements))

        else:
            save_result = False
            results = await get_results(CAS, client)
            display_results(results)

            if save_result:
                with open("results.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                 

if __name__ == "__main__":
    asyncio.run(demo())