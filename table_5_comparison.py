import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import measures
from agreement_matrix import file_to_graphs, generate_synthetic_dataset

"""
Comparaison des niveaux d'homophilie moyens sur les différents 
jeux de données (Section 6, Tableau 5 de l'article).
"""

# Calcule les moyennes et écart-types des caractéristiques du graphe et des différentes mesures d'homophilie
def compute_metrics(graphs):
    ns = [g.number_of_nodes() for g in graphs]
    es = [g.number_of_edges() for g in graphs]
    ms = [len(set(nx.get_node_attributes(g, 'label').values())) for g in graphs]

    h_edge = [measures.edge_homophily(g) for g in graphs]
    h_node = [measures.node_homophily(g) for g in graphs]
    h_class = [measures.class_homophily(g) for g in graphs]
    h_adj = [measures.adjusted_homophily(g) for g in graphs]
    h_unb = [measures.unbiased_homophily(g) for g in graphs]
    h_unb_alpha = [measures.unbiased_homophily_alpha(g, 0.5) for g in graphs]

    return {
        "n": np.mean(ns),
        "|E|": np.mean(es),
        "m": np.mean(ms),
        "h_edge": np.mean(h_edge),
        "h_edge_std": np.std(h_edge),
        "h_node": np.mean(h_node),
        "h_node_std": np.std(h_node),
        "h_class": np.mean(h_class),
        "h_class_std": np.std(h_class),
        "h_adj": np.mean(h_adj),
        "h_adj_std": np.std(h_adj),
        "h_unb": np.mean(h_unb),
        "h_unb_std": np.std(h_unb),
        "h_unb_a=0.5": np.mean(h_unb_alpha),
        "h_unb_a=0.5_std": np.std(h_unb_alpha),
    }

# Charge l'ensemble des jeux de données, calcule les statistiques comparatives et sauvegarde le tableau dans /results
def generate_table_5():
    datasets = {}

    # 1. NCI1
    if os.path.exists("nci1_pairs.pkl"):
        with open("nci1_pairs.pkl", "rb") as f:
            datasets["NCI1"] = pickle.load(f)["graphes"]

    # 2. MUTAG
    if os.path.exists("data/MUTAG"):
        datasets["MUTAG"] = list(file_to_graphs("data/MUTAG").values())

    # 3. PROTEINS
    if os.path.exists("data/PROTEINS"):
        datasets["PROTEINS"] = list(file_to_graphs("data/PROTEINS").values())

    # 4. SYNTHETIC SBM
    datasets["Synthetic (SBM)"] = generate_synthetic_dataset(n_graphs=300, seed=42)

    rows = []
    for name, graphs in datasets.items():
        stats = compute_metrics(graphs)
        rows.append({
            "Dataset": name,
            "Graphes": len(graphs),
            "n moy.": round(stats["n"], 1),
            "|E| moy.": round(stats["|E|"], 1),
            "m moy.": round(stats["m"], 2),
            "h_edge": f"{stats['h_edge']:.4f} (±{stats['h_edge_std']:.2f})",
            "h_node": f"{stats['h_node']:.4f} (±{stats['h_node_std']:.2f})",
            "h_class": f"{stats['h_class']:.4f} (±{stats['h_class_std']:.2f})",
            "h_adj": f"{stats['h_adj']:.4f} (±{stats['h_adj_std']:.2f})",
            "h_unb": f"{stats['h_unb']:.4f} (±{stats['h_unb_std']:.2f})",
            "h_unb (a=0.5)": f"{stats['h_unb_a=0.5']:.4f} (±{stats['h_unb_a=0.5_std']:.2f})",
        })

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/table_5.csv", index=False)
    print("Résultats sauvegardés dans : results/table_5.csv")
    return df

if __name__ == "__main__":
    generate_table_5()
