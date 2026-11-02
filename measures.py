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

def unbiased_homophily(G: nx.Graph, alpha : float): 

    labels = set(nx.get_node_attributes(G, "label").values())
    res = 0.0
    c_ii = {}
    c_ij = {}

    """ calcul des cii """
    for u,v in G.edges() : 
        i = G.nodes[u]["label"]
        j = G.nodes[v]["label"]

        if i == j : 
            if i in c_ii : 
                c_ii[i] += 1
            else : 
                c_ii[i] = 1

        else : 
            if (i,j) in c_ij : 
                c_ij[(i,j)] += 1
            else : 
                c_ij[(i,j)] = 1

    num_edges = G.number_of_edges()
    if num_edges == 0 : 
        return 0.0

    c_ii = {i : c_ii[i] / num_edges for i in c_ii}
    c_ij = {(i,j) : c_ij[(i,j)] / num_edges for (i,j) in c_ij}

    numerator = 0.0
    denominator = 0.0

    for i in labels : 
        for j in labels : 
            if i < j :
                numerator += math.sqrt(c_ii[i]*c_ii[j]) - c_ij[(i,j)]
                denominator += math.sqrt(c_ii[i]*c_ii[j]) + c_ij[(i,j)]

    part1 = (numerator/denominator) if denominator != 0 else 0.0

    sum_cii = sum(math.sqrt(c_ii[i] for i in labels))
    part2 = alpha * min(sum_cii,1)

    res = part1 + part2

    return res
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
