from sentence_transformers import SentenceTransformer
import numpy as np

modele_multi = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

CORPUS = [
    "Le triathlon est un sport d'endurance qui enchaîne trois disciplines dans l'ordre : natation, cyclisme et course à pied. Le chronomètre ne s'arrête jamais entre les épreuves, les transitions font partie intégrante de la course.",
    "Le format sprint comprend 750 m de natation, 20 km de vélo et 5 km de course à pied. C'est le format le plus accessible pour débuter, avec un effort total d'environ une heure pour les amateurs entraînés.",
    "La distance olympique, aussi appelée distance M ou courte distance, se compose de 1,5 km de natation, 40 km de vélo et 10 km de course à pied. C'est le format disputé aux Jeux olympiques depuis Sydney 2000.",
    "Le half-Ironman, ou Ironman 70.3, totalise 113 km : 1,9 km de natation, 90 km de vélo et un semi-marathon de 21,1 km. Le chiffre 70.3 correspond à la distance totale exprimée en miles.",
    "L'Ironman, ou triathlon XXL, est le format mythique de la longue distance : 3,8 km de natation, 180 km de vélo et un marathon complet de 42,195 km. Les meilleurs professionnels le bouclent en moins de 8 heures.",
    "Le championnat du monde Ironman s'est historiquement disputé à Kona, sur l'île d'Hawaï, depuis 1978. La course est réputée pour sa chaleur, son humidité et les vents violents de la traversée de l'Energy Lab.",
    "La transition T1 désigne le passage de la natation au vélo, et la T2 le passage du vélo à la course à pied. Une transition efficace peut faire gagner de précieuses secondes, on parle parfois de quatrième discipline.",
    "En triathlon courte distance au niveau élite, le drafting (aspiration-abri derrière un autre cycliste) est autorisé. Sur les épreuves longue distance et la plupart des courses amateurs, il est interdit et sanctionné par des pénalités.",
    "Le port de la combinaison néoprène en natation dépend de la température de l'eau : elle est généralement obligatoire en dessous de 16 °C et interdite au-dessus de 24,5 °C pour les groupes d'âge selon la réglementation.",
    "La natation en eau libre se déroule en lac, en mer ou en rivière. Les triathlètes utilisent la technique du water-polo, tête hors de l'eau, pour se repérer par rapport aux bouées du parcours.",
    "Le vélo de contre-la-montre, avec son prolongateur aéro, ses roues à jantes hautes et sa géométrie spécifique, est privilégié sur les formats longs sans drafting. Sur courte distance avec drafting, le vélo de route classique est obligatoire.",
    "L'enchaînement vélo-course provoque une sensation de jambes lourdes bien connue des triathlètes. Les entraînements spécifiques appelés briques, qui enchaînent les deux disciplines, permettent d'habituer le corps à cette transition.",
    "La nutrition est souvent considérée comme le nerf de la guerre sur longue distance. Sur un Ironman, un athlète vise généralement entre 60 et 90 grammes de glucides par heure sur le vélo, via gels, boissons et barres énergétiques.",
    "Les élastiques de chaussures et les pédales automatiques permettent de gagner du temps en transition : les triathlètes laissent leurs chaussures clipsées sur le vélo et les enfilent en roulant après un départ pieds nus.",
    "Le triathlon est né dans les années 1970 en Californie, à San Diego, avant de connaître son essor avec le premier Ironman d'Hawaï en 1978, remporté par Gordon Haller devant 14 autres participants.",
    "La Norvège a dominé le triathlon mondial ces dernières années, avec Kristian Blummenfelt, champion olympique à Tokyo et champion du monde Ironman, et Gustav Iden, vainqueur à Kona en 2022 avec un record du parcours.",
    "Le Norseman, en Norvège, est réputé comme l'un des triathlons extrêmes les plus durs au monde : départ natation depuis un ferry dans un fjord glacial, 180 km de vélo en montagne et une arrivée au sommet du Gaustatoppen.",
    "En course par équipe, le relais mixte est devenu discipline olympique à Tokyo 2020 : deux femmes et deux hommes enchaînent chacun un super-sprint d'environ 300 m de natation, 8 km de vélo et 2 km de course.",
    "La fréquence cardiaque, la puissance en watts sur le vélo et l'allure au kilomètre en course à pied sont les trois métriques principales utilisées par les triathlètes pour gérer leur intensité et éviter l'explosion en fin d'épreuve.",
    "Les catégories d'âge, appelées groupes d'âge ou age groups, permettent aux amateurs de se comparer par tranches de cinq ans. Les meilleurs de chaque catégorie peuvent se qualifier pour les championnats du monde, comme les slots pour Kona.",
]

# QUESTIONS[i] doit idéalement remonter CORPUS[i] en premier
QUESTIONS = [
    "Quelles sont les trois disciplines enchaînées lors d'un triathlon ?",
    "Quel format de triathlon est le plus adapté pour un débutant ?",
    "Quelles sont les distances du triathlon aux Jeux olympiques ?",
    "Pourquoi le half-Ironman s'appelle-t-il 70.3 ?",
    "Quelle est la longueur totale d'un Ironman ?",
    "Où se déroule historiquement le championnat du monde Ironman ?",
    "Que signifient T1 et T2 en triathlon ?",
    "Est-ce qu'on a le droit de rouler dans la roue d'un autre concurrent ?",
    "À partir de quelle température d'eau la combinaison est-elle interdite ?",
    "Comment fait-on pour s'orienter quand on nage en eau libre ?",
    "Quel type de vélo choisir pour un triathlon longue distance ?",
    "C'est quoi un entraînement en brique ?",
    "Combien de glucides faut-il consommer par heure sur un Ironman ?",
    "Quelles astuces matérielles permettent d'accélérer le passage en transition ?",
    "Dans quelle ville le triathlon a-t-il été inventé ?",
    "Qui sont les triathlètes norvégiens les plus titrés ?",
    "Quel est le triathlon extrême avec un départ depuis un ferry dans un fjord ?",
    "Comment fonctionne le relais mixte olympique en triathlon ?",
    "Quelles métriques utiliser pour bien doser son effort en course ?",
    "Comment les amateurs peuvent-ils se qualifier pour les championnats du monde ?",
]

query = "comment gérer son alimentation pendant une longue course"
mat = modele_multi.encode(CORPUS)

def cosinus(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search(query: str, corpus: list[str], mat, k: int = 3) -> list[tuple[float, str]]:
    q = modele_multi.encode(query)
    scores = []
    for i, vector in enumerate(mat):
        score = cosinus(q, vector)
        scores.append((score, corpus[i]))
    return sorted(scores, reverse=True)[:k]

def recall_at(questions: list[str], corpus: list[str], mat, k: int) -> tuple[float, list[tuple[str, str]]]:
    ok = 0
    echec = []
    for i, question in enumerate(questions):
        top = search(question, corpus, mat, k=k)
        if any(texte == corpus[i] for _, texte in top):
            ok += 1
        else:
            echec.append((question, top))
    return (ok/len(questions), echec)

for k in [1,3]:
    print(f"recall@{k} : {recall_at(QUESTIONS, CORPUS, mat, k=k)}")