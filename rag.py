import dataclasses

from llm_client import GEMINI_API_KEY, LLMClient, ModelInfo, Stats, stats
from retrieval import CORPUS, QUESTIONS, mat, search_reranked
import asyncio
import json
import sys

""" ---------------- globals ---------------- """

GEN_VERSION = "v1"
JUDGE_PROMPT_VERSION = "v2"
CAS = json.load(open("eval_set.json", encoding="utf-8"))
human_jugement = json.load(open("labels_humains.json", encoding="utf-8")) 

""" ---------------- prompts ---------------- """

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


def prompt_builder(documents, question):
    p = f"""
    Réponds à la question en te basant uniquement sur les documents ci-dessous.
    Si l'information n'y figure pas, dis-le.

    Documents :
    {documents}

    Question : {question}"""
    return p

""" ---------------- eval ---------------- """

def evaluate(cas: dict, texte: str, sources: list[int], documents: str) -> dict:
    if texte is None:
        texte = "error"
        retrieval = False
        keywords = False
        not_empty = False
    else:
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

""" ---------------- judge ---------------- """

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


def prepare_judge(result: dict) -> str: # prompt
    prompt = judge_prompt(result["documents"], result["question"], result["texte"])
    return prompt


async def juge(results: list[dict], client: LLMClient) -> list[dict]:
    preps_judge = [prepare_judge(r) for r in results]
    calls = await client.complete_many(preps_judge)
    judgements = []
    for call in calls:
        jugement = call.text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(jugement)
        except json.JSONDecodeError:
            judgements.append({"verdict": None, "raison": "parsing failed", "brut": call.text})
            continue

        judge_response = {
            "verdict": {"oui": True, "non": False}.get(data.get("verdict")),
            "raison": data.get("raison"),
            "brut": None
        }
        judgements.append(judge_response)
    return judgements

""" ---------------- results ---------------- """

def prepare(question: str) -> tuple[str, list[int], str]: # prompt, sources, documents
    top = search_reranked(question, CORPUS, mat, k=3)
    documents = "\n".join(f"[{i}] {doc}" for i, (_, _, doc) in enumerate(top, 1))
    prompt = prompt_builder(documents, question)
    return (prompt, [idx for _, idx, _ in top], documents)


async def get_results(cas, client: LLMClient) -> tuple[list[dict], Stats]:
    preps = [prepare(c["question"]) for c in cas]
    calls = await client.complete_many([p for p, _, _ in preps])
    statistics = stats(calls)
    return ([evaluate(c, call.text, src, docs)
        for c, call, (_, src, docs) in zip(cas, calls, preps, strict=True)], statistics)

""" ---------------- display ---------------- """

def display_dict(title: str | None, d: dict):
    if title is not None:
        print(f"\n{title}")
    for cle, valeur in d.items():
        print(f"  {cle:<20} {valeur}")


def display_results(results):
    n = len(results)
    for i, c in enumerate(results):
        print(f"[{i}] Question: {c['question']}\nResponse: {c['texte']}\nSource: {c['sources']}\n\n")
    print(f"Retrieval : {sum(1 for r in results if r['retrieval']) / n:.0%}")
    print(f"Keywords  : {sum(1 for r in results if r['keywords']) / n:.0%}")
    print(f"Not empty  : {sum(1 for r in results if r['not_empty']) / n:.0%}")


def display_jugement(results: list[dict], judgements: dict): 
    print(f"\nJugement:")
    for i, (res, judgement) in enumerate(zip(results, judgements)):
        print(f"Question: {res['question']}")
        print(f"Response: {res['texte']}")
        print(f"Verdict juge vs human: {judgement['verdict']} | {human_jugement[i]['info']}")
        print(f"Raison: {judgement['raison']}")


def display_comparaison(comp: dict, total_size: int):
    print(f"\nComparaison (total: {total_size}):")
    print(f"{comp['detected_hallu']} hallucination detected, " 
          f"{comp['missed_hallu']} hallucination missed, "
          f"{comp['false_alert']} false alert, "
          f"{comp['agree']} agreed, "
          f"{comp['failed_judgement']} failed judgements"
          f"\n{((comp['detected_hallu'] + comp['agree']) / total_size):.0%} overall agreement rate")


""" ---------------- start point ---------------- """

async def demo():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    # "generate" | "judge" | "compare"

    gem_3_1_flash_lite = ModelInfo("gemini-3.1-flash-lite", 0.25, 1.50)
    
    async with LLMClient(model=gem_3_1_flash_lite, api_key=GEMINI_API_KEY) as client:
        if mode == "generate":
            results, statistics = await get_results(CAS, client)
            display_results(results)
            display_dict("Statistics", dataclasses.asdict(statistics))

            with open(f"{GEN_VERSION}_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        elif mode == "judge":
            try:
                loaded_results = json.load(open(f"{GEN_VERSION}_results.json", encoding="utf-8"))
            except FileNotFoundError:
                print(f"Error : {GEN_VERSION}_results.json not found : python rag.py generate")
                return
            
            judgements = await juge(results=loaded_results, client=client)
            display_jugement(loaded_results, judgements)

            with open(f"{JUDGE_PROMPT_VERSION}_judged.json", "w", encoding="utf-8") as f:
                json.dump(judgements, f, ensure_ascii=False, indent=2)

        elif mode == "compare":
            # tofix: labels_humains.json with version 
            # {GEN_VERSION}_{JUDGE_PROMPT_VERSION}_judged.json for compatibles versions
            judgements = json.load(open(f"{JUDGE_PROMPT_VERSION}_judged.json", encoding="utf-8"))
            comp = compare_judgements(judgements, human_jugement)
            display_comparaison(comp, len(judgements))

        else:
            print("Invalid mode")
            return
   

if __name__ == "__main__":
    asyncio.run(demo())