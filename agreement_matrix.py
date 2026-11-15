import os
import random
import pickle
from itertools import combinations
import numpy as np
import networkx as nx
import pandas as pd
import measures

"""
Calcul des matrices d'accord complètes pour évaluer la cohérence 
des mesures d'homophilie sur différents datasets (Section 3 de l'article).
"""

OUTPUT_DIR = "results"

MEASURES = {
    "Edge": measures.edge_homophily,
    "Node": measures.node_homophily,
    "Class": measures.class_homophily,
    "Adjusted": measures.adjusted_homophily,
    "Unbiased (alpha=0)": measures.unbiased_homophily,
    "Unbiased (alpha=0.5)": lambda G: measures.unbiased_homophily_alpha(G, alpha=0.5),
}

# Charge les graphes d'un dataset à partir des fichiers texte (MUTAG, PROTEINS)
def file_to_graphs(file_path):
    graphs = {}
    dataset_name = os.path.basename(os.path.normpath(file_path))
    files_source_path = os.path.join(file_path, dataset_name)

    node_labels = {}
    with open(files_source_path + "_node_labels.txt", "r", encoding="utf-8") as file:
        for i, label in enumerate(file, start=1):
            node_labels[i] = int(label.strip())

    node_graph = {}
    with open(files_source_path + "_graph_indicator.txt", "r", encoding="utf-8") as file:
        for i, graph_id in enumerate(file, start=1):
            graph_id = graph_id.strip()
            if graph_id not in graphs:
                graphs[graph_id] = nx.Graph(name=graph_id)
            node_graph[i] = graph_id
            graphs[graph_id].add_node(str(i), label=node_labels[i])

    with open(files_source_path + "_A.txt", "r", encoding="utf-8") as file:
        for edge in file:
            nodes = edge.strip().split(",")
            u = nodes[0].strip()
            v = nodes[1].strip()
            u_id = int(u)
            v_id = int(v)
            graph_id = node_graph[u_id]
            if node_graph[v_id] != graph_id:
                raise ValueError(f"Arête entre deux graphes différents : {u}, {v}")
            graphs[graph_id].add_edge(u, v)

    return graphs

# Charge le dataset NCI1 depuis le cache pickle ou le prépare avec PyTorch Geometric
def load_nci1():
    pkl_path = "nci1_pairs.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data["graphes"], data["paires"]

    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx

    dataset = TUDataset(root="data/NCI1", name="NCI1")
    graphs = []
    for data in dataset:
        if data.x is not None:
            G = to_networkx(data, to_undirected=True)
            for node in G.nodes():
                G.nodes[node]["label"] = int(data.x[node].argmax().item())
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                graphs.append(G)

    pairs = get_or_sample_pairs(graphs, n_pairs=1000, seed=42)
    with open(pkl_path, "wb") as f:
        pickle.dump({"graphes": graphs, "paires": pairs}, f)

    return graphs, pairs

# Génère un graphe synthétique via le modèle à blocs stochastiques (SBM)
def generate_sbm_graph(n_nodes=100, n_classes=3, imbalance=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if imbalance:
        probs = rng.dirichlet(np.ones(n_classes) * 0.4)
    else:
        probs = rng.dirichlet(np.ones(n_classes) * 5.0)

    sizes = rng.multinomial(n_nodes, probs)
    sizes = [max(s, 2) for s in sizes]

    p_in = rng.uniform(0.05, 0.35)
    p_out = rng.uniform(0.02, 0.30)

    p_matrix = np.full((n_classes, n_classes), p_out)
    np.fill_diagonal(p_matrix, p_in)

    G = nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=int(rng.integers(1e9)))

    node_idx = 0
    for class_id, size in enumerate(sizes):
        for _ in range(size):
            G.nodes[node_idx]["label"] = class_id
            node_idx += 1

    return G

# Génère une collection de graphes synthétiques SBM variés
def generate_synthetic_dataset(n_graphs=300, seed=42):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n_graphs):
        n_nodes = int(rng.integers(50, 150))
        n_classes = int(rng.integers(2, 6))
        imbalance = bool(rng.choice([True, False]))
        G = generate_sbm_graph(n_nodes=n_nodes, n_classes=n_classes, imbalance=imbalance, rng=rng)
        if G.number_of_edges() > 0:
            graphs.append(G)
    return graphs

# Calcule la matrice d'accord en pourcentage entre les différentes mesures d'homophilie
def compute_agreement(graphs, pairs, measures_dict, tolerance=1e-12):
    names = list(measures_dict.keys())
    scores = {name: [func(G) for G in graphs] for name, func in measures_dict.items()}

    decisions = {}
    for name in names:
        dec = []
        for i, j in pairs:
            diff = scores[name][i] - scores[name][j]
            if abs(diff) <= tolerance:
                dec.append(0)
            elif diff > 0:
                dec.append(1)
            else:
                dec.append(-1)
        decisions[name] = dec

    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    nb_pairs = len(pairs)
    for m1 in names:
        for m2 in names:
            agreements = sum(a == b for a, b in zip(decisions[m1], decisions[m2]))
            matrix.loc[m1, m2] = 100.0 * agreements / nb_pairs

    return matrix.round(2)

# Échantillonne aléatoirement un nombre fixé de paires de graphes
def get_or_sample_pairs(graphs, n_pairs=1000, seed=42):
    all_pairs = list(combinations(range(len(graphs)), 2))
    rng = random.Random(seed)
    return rng.sample(all_pairs, min(n_pairs, len(all_pairs)))

# Gère les paires, calcule la matrice d'accord puis la sauvegarde dans /results
def process_dataset(name, graphs, pairs=None, filename=None):
    if pairs is None:
        pairs = get_or_sample_pairs(graphs)

    matrix = compute_agreement(graphs, pairs, MEASURES)

    if filename:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, filename)
        matrix.to_csv(file_path)
        print(f"Matrice sauvegardée dans : {file_path}")

    return matrix

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nci1_graphs, nci1_pairs = load_nci1()
    process_dataset("NCI1", nci1_graphs, nci1_pairs, filename="agreement_matrix_nci1.csv")

    if os.path.exists("data/MUTAG"):
        mutag_dict = file_to_graphs("data/MUTAG")
        mutag_graphs = list(mutag_dict.values())
        process_dataset("MUTAG", mutag_graphs, filename="agreement_matrix_mutag.csv")

    if os.path.exists("data/PROTEINS"):
        proteins_dict = file_to_graphs("data/PROTEINS")
        proteins_graphs = list(proteins_dict.values())
        process_dataset("PROTEINS", proteins_graphs, filename="agreement_matrix_proteins.csv")

    synthetic_graphs = generate_synthetic_dataset(n_graphs=300, seed=42)
    process_dataset("Synthetic_SBM", synthetic_graphs, filename="agreement_matrix_synthetic.csv")
