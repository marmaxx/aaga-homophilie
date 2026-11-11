import networkx as nx
import math

def edge_homophily(G: nx.Graph):
    if G.number_of_edges() == 0:
        return 0.0
    res = 0.0
    for u,v in G.edges:
        if G.nodes[u]["label"] == G.nodes[v]["label"]:
            res += 1
    return res / G.number_of_edges()

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

def adjusted_homophily(G: nx.Graph):
    if G.number_of_edges() == 0:
        return 0.0
    labels = set(nx.get_node_attributes(G, "label").values())
    if len(labels) < 2 or G.number_of_nodes() == 0:
            return 0.0
    s = 0.0
    for l in labels:
        label_degree = sum(G.degree(v) for v in G.nodes() if G.nodes[v]["label"] == l)
        s += (label_degree  / (2 * G.number_of_edges())) ** 2
    if s == 1:
        return 0.0
    return (edge_homophily(G) - s) / (1 - s)

def unbiased_homophily_alpha(G: nx.Graph, alpha: float):
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
    c_ij = {pair: value / m for pair, value in c_ij.items()}

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

    denominator = (sum_sqrt ** 2 + 1 - 2 * sum(c_ii.values()))

    return numerator / denominator if denominator != 0 else 0.0

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


import os
import networkx as nx


def file_to_graphs(file_path):
    graphs = {}

    dataset_name = os.path.basename(os.path.normpath(file_path))
    files_source_path = os.path.join(file_path, dataset_name)

    node_labels = {}

    with open(
        files_source_path + "_node_labels.txt",
        "r",
        encoding="utf-8"
    ) as file:

        for i, label in enumerate(file, start=1):
            node_labels[i] = int(label.strip())

    node_graph = {}

    with open(
        files_source_path + "_graph_indicator.txt",
        "r",
        encoding="utf-8"
    ) as file:

        for i, graph_id in enumerate(file, start=1):

            graph_id = graph_id.strip()

            if graph_id not in graphs:
                graphs[graph_id] = nx.Graph(name=graph_id)

            node_graph[i] = graph_id

            graphs[graph_id].add_node(
                str(i),
                label=node_labels[i]
            )

    with open(
        files_source_path + "_A.txt",
        "r",
        encoding="utf-8"
    ) as file:

        for edge in file:

            nodes = edge.strip().split(",")

            u = nodes[0].strip()
            v = nodes[1].strip()

            u_id = int(u)
            v_id = int(v)

            graph_id = node_graph[u_id]

            if node_graph[v_id] != graph_id:
                raise ValueError(
                    f"Arête entre deux graphes différents : {u}, {v}"
                )

            graphs[graph_id].add_edge(u, v)

    return graphs

Graphs_of_Mutag = file_to_graphs("datasets/MUTAG")

Graphs_of_Proteins = file_to_graphs("datasets/PROTEINS")


print("Nombre de graphes MUTAG :", len(Graphs_of_Mutag))
print("Nombre de graphes PROTEINS :", len(Graphs_of_Proteins))

for graph_id, G in list(Graphs_of_Mutag.items())[:3]:

    print(
        f"Graphe {graph_id} : "
        f"{G.number_of_nodes()} nœuds, "
        f"{G.number_of_edges()} arêtes"
    )


Graphs_of_Mutag = file_to_graphs("datasets/MUTAG")
Graphs_of_Proteins = file_to_graphs("datasets/PROTEINS")

measures = {
    "edge": edge_homophily,
    "node": node_homophily,
    "class": class_homophily,
    "adjusted": adjusted_homophily,
    "unbiased_homophily" : unbiased_homophily
}

for name, measure in measures.items():
    print(name, measure(G1), measure(G2))

alpha = 0.5

print("unbiased_homophily_alpha",
      unbiased_homophily_alpha(G1, alpha),
      unbiased_homophily_alpha(G2, alpha))
