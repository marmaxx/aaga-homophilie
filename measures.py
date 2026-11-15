import os
import math
import networkx as nx

"""
Implémentation des 6 différents mesures d'homophilie présentes dans l'article.
"""

# Mesure d'homophilie des arêtes
def edge_homophily(G: nx.Graph):
    if G.number_of_edges() == 0:
        return 0.0
    res = 0.0
    for u, v in G.edges:
        if G.nodes[u]["label"] == G.nodes[v]["label"]:
            res += 1
    return res / G.number_of_edges()

# Mesure d'homophilie des noeuds
def node_homophily(G: nx.Graph):
    res = 0.0
    count = 0
    for v in G.nodes:
        neighbors = list(G.neighbors(v))
        if len(neighbors) != 0:
            label = G.nodes[v]["label"]
            res += sum(G.nodes[n]["label"] == label for n in neighbors) / len(neighbors)
            count += 1
    if G.number_of_nodes() == 0:
        return 0.0
    return res / count if count != 0 else 0.0

# Mesure d'homophilie des classes
def class_homophily(G: nx.Graph):
    labels = set(nx.get_node_attributes(G, "label").values())
    if len(labels) < 2 or G.number_of_nodes() == 0:
        return 0.0
    n = G.number_of_nodes()
    res = 0.0
    for l in labels:
        label_size = 0
        label_degree = 0
        s = 0
        for v in G.nodes():
            if G.nodes[v]["label"] == l:
                label_size += 1
                neighbors = list(G.neighbors(v))
                label_degree += len(neighbors)
                s += sum(G.nodes[neighbor]["label"] == l for neighbor in neighbors)
        if label_degree > 0:
            res += max(s / label_degree - label_size / n, 0)
    return res / (len(labels) - 1)

# Mesure d'homophilie ajustée
def adjusted_homophily(G: nx.Graph):
    if G.number_of_edges() == 0:
        return 0.0
    labels = set(nx.get_node_attributes(G, "label").values())
    if len(labels) < 2 or G.number_of_nodes() == 0:
        return 0.0
    s = 0.0
    for l in labels:
        label_degree = sum(G.degree(v) for v in G.nodes() if G.nodes[v]["label"] == l)
        s += (label_degree / (2 * G.number_of_edges())) ** 2
    if s == 1:
        return 0.0
    return (edge_homophily(G) - s) / (1 - s)

# Mesure d'homophilie non biaisée avec paramètre alpha
def unbiased_homophily_alpha(G: nx.Graph, alpha: float = 0.0):
    labels = sorted(set(nx.get_node_attributes(G, "label").values()))

    if len(labels) < 2 or G.number_of_edges() == 0:
        return 0.0

    c_ii = {}
    c_ij = {}

    for u, v in G.edges():
        i = G.nodes[u]["label"]
        j = G.nodes[v]["label"]

        if i == j:
            c_ii[i] = c_ii.get(i, 0) + 1
        else:
            pair = tuple(sorted((i, j)))
            c_ij[pair] = c_ij.get(pair, 0) + 1

    m = G.number_of_edges()

    c_ii = {i: c_ii.get(i, 0) / m for i in labels}
    c_ij = {pair: value / (2 * m) for pair, value in c_ij.items()}

    numerator = 0.0
    denominator = 0.0

    for index, i in enumerate(labels):
        for j in labels[index + 1:]:
            cross = c_ij.get((i, j), 0)
            product = math.sqrt(c_ii[i] * c_ii[j])
            numerator += product - cross
            denominator += product + cross

    part1 = numerator / denominator if denominator != 0 else 0.0

    sum_cii = sum(math.sqrt(c_ii[i]) for i in labels)
    part2 = alpha * min(sum_cii, 1)

    return part1 + part2

# Mesure d'homophilie non biaisée 
def unbiased_homophily(G: nx.Graph):
    labels = set(nx.get_node_attributes(G, "label").values())
    m = G.number_of_edges()

    if len(labels) < 2 or m == 0:
        return 0.0

    c_ii = {i: 0 for i in labels}

    for u, v in G.edges():
        i = G.nodes[u]["label"]
        j = G.nodes[v]["label"]
        if i == j:
            c_ii[i] += 1

    c_ii = {i: c_ii[i] / m for i in labels}
    sum_sqrt = sum(math.sqrt(c_ii[i]) for i in labels)

    numerator = sum_sqrt ** 2 - 1
    denominator = sum_sqrt ** 2 + 1 - 2 * sum(c_ii.values())

    if denominator == 0:
        return 0.0

    return numerator / denominator

# Tests des mesures sur deux graphes simples G1 et G2
if __name__ == "__main__":
    G1 = nx.Graph()
    G1.add_nodes_from([
        ("A", {"label": 0}),
        ("B", {"label": 0}),
        ("C", {"label": 1}),
        ("D", {"label": 1}),
    ])
    G1.add_edges_from([
        ("A", "B"),
        ("C", "D"),
        ("B", "C"),
    ])

    G2 = nx.Graph()
    G2.add_nodes_from([
        ("A", {"label": 0}),
        ("B", {"label": 0}),
        ("C", {"label": 1}),
        ("D", {"label": 1}),
    ])
    G2.add_edges_from([
        ("A", "C"),
        ("A", "B"),
        ("B", "D"),
    ])

    test_measures = {
        "edge": edge_homophily,
        "node": node_homophily,
        "class": class_homophily,
        "adjusted": adjusted_homophily,
        "unbiased_homophily": unbiased_homophily,
    }

    print("Test sur G1 et G2 :")
    for name, measure in test_measures.items():
        print(f"  {name:20s}  G1: {measure(G1):.4f} | G2: {measure(G2):.4f}")

    alpha = 0.5
    print("  unbiased_alpha (0.5)  G1:", round(unbiased_homophily_alpha(G1, alpha), 4),
          "| G2:", round(unbiased_homophily_alpha(G2, alpha), 4))
    print("  unbiased_alpha (0.0)  G1:", round(unbiased_homophily_alpha(G1, 0.0), 4),
          "| G2:", round(unbiased_homophily_alpha(G2, 0.0), 4))
