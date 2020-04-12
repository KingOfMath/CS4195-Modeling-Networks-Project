import networkx as nx


def parseData(path):
    file = open(path)
    data = []
    line = file.readline()
    while line:
        data.append(line)
        line = file.readline()
    file.close()
    return data


def raws_to_tuple(raws):
    tuples = []
    for r in raws:
        tuples.append((int(r.split()[0]), int(r.split()[1]), int(r.split()[2]), int(r.split()[3])))
    return tuples


def generate_bipartite_graph(raws_file=None):
    if raws_file is not None:
        print("[NETWORK INIT] Read network from sources/" + raws_file)
        raws = parseData("../../sources/" + raws_file)
        raws = raws_to_tuple(raws)
        G = nx.Graph()
        for e in raws:
            G.add_node(e[0], bipartite=0)
            G.add_node(e[1] + 943, bipartite=1)
            G.add_edge(e[0], e[1] + 943, weight=e[2], timestamp=e[3])
        print("[NETWORK INIT] Bipartite Graph generate")
    else:
        print("[ERROR] Specify source file")
    return G
