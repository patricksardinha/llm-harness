from sentence_transformers import SentenceTransformer
import numpy as np

modele_en = SentenceTransformer("all-MiniLM-L6-v2")
modele_multi = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

chat = modele_multi.encode("chat")
chien = modele_multi.encode("chien")
felin = modele_multi.encode("félin")
voiture = modele_multi.encode("voiture")
BZ_4471_A = modele_multi.encode("BZ-4471-A")
BZ_4471_B = modele_multi.encode("BZ-4471-B")

def cosinus(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(cosinus(chat, felin))
print(cosinus(chat, voiture))
print(cosinus(BZ_4471_A, BZ_4471_B))
