
import random
import pickle
from itertools import combinations

import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx


SEED = 42
NB_PAIRS = 1000

dataset = TUDataset(root="data/NCI1", name="NCI1")

graphs = []

for i, data in enumerate(dataset):
    if data.x is None:
        raise ValueError(
            f"Le graphe {i} ne contient pas de caractéristiques de nœuds."
        )

    G = to_networkx(data, to_undirected=True)

    for node in G.nodes():
        G.nodes[node]["label"] = int(data.x[node].argmax().item())

    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        graphs.append(G)

print("Nombre de graphes chargés :", len(graphs))

all_pairs = list(combinations(range(len(graphs)), 2))

if NB_PAIRS > len(all_pairs):
    raise ValueError("Nombre de paires demandé trop élevé.")

rng = random.Random(SEED)
pairs = rng.sample(all_pairs, NB_PAIRS)

with open("nci1_pairs.pkl", "wb") as f:
    pickle.dump(
        {"graphes": graphs, "paires": pairs},
        f
    )

print(f"{len(pairs)} paires sauvegardées dans nci1_pairs.pkl")
