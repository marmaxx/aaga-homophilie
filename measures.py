import networkx as nx

def edge_homophily(G: nx.Graph):
    res = 0.0
    for u,v in G.edges:
        if G.nodes[u]["label"] == G.nodes[v]["label"]:
            res += 1
    return res / G.number_of_edges()

def node_homophily(G: nx.Graph):
    res = 0.0
    for v in G.nodes:
        neighbors = list(G.neighbors(v))
        if len(neighbors) != 0:
            label = G.nodes[v]["label"]
            res += sum(G.nodes[n]["label"] == label for n in neighbors) / len(neighbors)
    if G.number_of_nodes() == 0:
        return 0.0
    return res / G.number_of_nodes()

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
        for v in G.nodes:
            if G.nodes[v]["label"] == l:
                label_size += 1
                neighbors = list(G.neighbors(v))
                label_degree += len(neighbors)
                s += sum(G.nodes[neighbor]["label"] == l for neighbor in neighbors)
        if label_degree > 0:
            res += max(s / label_degree - label_size / n, 0)
    return res / (len(labels) - 1)

def adjusted_homophily(G: nx.Graph):
    labels = set(nx.get_node_attributes(G, "label").values())
    if len(labels) < 2 or G.number_of_nodes() == 0:
            return 0.0
    s = 0.0
    for l in labels:
        label_degree = 0
        for v in G.nodes:
            if G.nodes[v]["label"] == l:
                label_degree += G.degree(v)
        s += (label_degree  / (2 * G.number_of_edges())) ** 2
    if s == 1:
        return 0.0
    return (edge_homophily(G) - s) / (1 - s)

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

for name, measure in {
    "edge": edge_homophily,
    "node": node_homophily,
    "class": class_homophily,
    "adjusted": adjusted_homophily,
}.items():
    print(name, measure(G1), measure(G2))