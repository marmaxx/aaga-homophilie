import pickle
import pandas as pd
import measures

measures = {
    "Edge": measures.edge_homophily,
    "Node": measures.node_homophily,
    "Class": measures.class_homophily,
    "Adjusted": measures.adjusted_homophily,
}

def compare(G1, G2, mesure, tolerance=1e-12):
    h1 = mesure(G1)
    h2 = mesure(G2)

    if abs(h1 - h2) <= tolerance:
        return 0  
    elif h1 > h2:
        return 1 
    else:
        return -1 
    
with open("nci1_pairs.pkl", "rb") as fichier:
    data = pickle.load(fichier)

graphs = data["graphes"]
pairs = data["paires"]

print(f"{len(graphs)} graphes chargés.")
print(f"{len(pairs)} paires chargées.")

decisions = {}

for name, measure in measures.items():
    decisions[name] = [
        compare(graphs[i], graphs[j], measure)
        for i, j in pairs
    ]

names = list(measures.keys())
matrix = pd.DataFrame(
    index=names,
    columns=names,
    dtype=float
)

nb_paires = len(pairs)

for measure_a in names:
    for measure_b in names:
        agreements = sum(
            a == b
            for a, b in zip(
                decisions[measure_a],
                decisions[measure_b]
            )
        )

        matrix.loc[measure_a, measure_b] = (
            100 * agreements / nb_paires
        )

# Afficher et sauvegarder la matrice
print("\nMatrice d'accord (en %) :")
print(matrix.round(2))

matrix.round(2).to_csv("agreement_matrix.csv")

print("\nMatrice sauvegardée dans agreement_matrix.csv")