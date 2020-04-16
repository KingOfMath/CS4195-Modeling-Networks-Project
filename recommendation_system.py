import networkx as nx
from util import generate_bipartite_graph
from networkx.algorithms import bipartite
from collections import Counter
from math import log2
import numpy as np
from scipy.spatial.distance import cosine as cos_dist
import progressbar
import itertools
import operator


def get_column(i, R):
    return [row[i] for row in R]


def vertical_padding(m, n, M):
    for i in range(0, m - n):
        pad = np.zeros((n + 1 + i, n))
        pad[:-1, :] = M
        M = pad
    return M


class RS:
    def __init__(self, G):
        self.G = G
        self.users, self.items = bipartite.sets(self.G)
        self.R = bipartite.matrix.biadjacency_matrix(self.G, self.items).toarray().tolist()
        self.U, self.S, self.V = np.linalg.svd(self.R)
        self.R_abs_i = None
        self.target = None

    def info(self):
        print()
        print("---- SYSTEM INFO ----")
        print("Bipartite: " + str(nx.is_connected(self.G)))
        print(
            "Users: " + str(len(self.users)) + " labels: [" + str(min(self.users)) + ", " + str(max(self.users)) + "]")
        print(
            "Items: " + str(len(self.items)) + " labels: [" + str(min(self.items)) + ", " + str(max(self.items)) + "]")
        print("R: (" + str(len(self.R)) + ", " + str(len(self.R[0])) + ") raws: [" + str(min(self.items)) +
              ", " + str(max(self.items)) + "] cols: [" + str(min(self.users)) + ", " + str(max(self.users)) + "]")
        print("U: " + str(self.U.shape))
        print("V: " + str(self.V.shape))
        print("S: " + str(self.S.shape))
        print()

    def commons_items(self, node_i, node_j):
        l = [x for x in nx.common_neighbors(self.G, node_i, node_j)]
        l.sort()
        return l

    def set_target_user(self, user):
        self.target = user
        self.abs_difference_rating_matr(self.target)

    def vectorize_users(self, user, dim=4):
        return get_column(user - 1, self.V)[0:dim]

    def abs_difference_rating_matr(self, user_i):
        self.R_abs_i = {u: {} for u in self.users if u != user_i}
        for user_j in self.users:
            if user_j != user_i:
                commons = self.commons_items(user_i, user_j)
                dict = {}
                for c in commons:
                    dict[c] = abs(self.R[c - min(self.items)][user_i - 1] - self.R[c - min(self.items)][user_j - 1])
                self.R_abs_i[user_j] = dict

    def count_abs_rating_difference(self, user_j):
        dict = {}
        for i in range(5):
            count = Counter(self.R_abs_i[user_j].values())[i]
            dict[i] = count
        return dict

    def weighted_sum(self, user_i):
        dict = {}
        for user_j in self.users:
            if user_j != user_i:
                count = self.count_abs_rating_difference(user_j)
                dict[user_j] = 1 * count[0] + 0.8 * count[1] + 0.6 * count[2] + 0.4 * count[3] + 0.2 * count[4]
        return dict

    def avg_rate(self, user_id):
        rates = [x for x in get_column(user_id - 1, self.R) if x != 0]
        return sum(rates) / len(rates)

    def de(self, user_i, user_j, item):
        return abs(
            (self.R[item - min(self.items)][user_i - 1] - self.avg_rate(user_i)) - (
                    self.R[item - min(self.items)][user_j - 1] - self.avg_rate(user_j)))

    def deviation(self, user_i, user_j, I):
        commons = self.commons_items(user_i, user_j)
        s = 0
        for i in range(0, I):
            s += self.de(user_i, user_j, commons[i])
        if s != 0:
            return s

    def probability_function(self, user_i, user_j, item_k, I):
        return self.de(user_i, user_j, item_k) / self.deviation(user_i, user_j, I)

    def entropy(self, user_i, user_j, I):
        commons = self.commons_items(user_i, user_j)
        H = 0
        for i in range(0, I):
            pk = self.probability_function(user_i, user_j, commons[i], I)
            H += pk * log2(pk)
        return -H

    def simE(self, user_i, user_j, I):
        return 1 - (self.entropy(user_i, user_j, I) / log2(I))

    def simC(self, user_i, user_j):
        ui = self.vectorize_users(user_i)
        uj = self.vectorize_users(user_j)
        sim = 1 - cos_dist(ui, uj)
        return sim

    def hybrid_similarity(self, user_i, user_j, I, beta, verbose):
        sim_c = self.simC(user_i, user_j)
        sim_e = self.simE(user_i, user_j, I)
        if verbose:
            print("Cosine: " + str(sim_c))
            print("Entropy: " + str(sim_e))
        return (beta * sim_c) + ((1 - beta) * sim_e)

    def compute_similar_users(self, thresh, beta, nodes=None, verbose=True):
        Wsum = self.weighted_sum(self.target)
        similar_users = []
        similarty_values = []
        if nodes is None:
            users_to_compute = self.users
        else:
            users_to_compute = nodes
        bar = progressbar.ProgressBar()
        for j in bar(users_to_compute):
            try:
                if j != self.target:
                    I = int(Wsum[j])
                    if I > 1:
                        hs = self.hybrid_similarity(self.target, j, I, beta, verbose)
                        if verbose:
                            print("Similarity(" + str(self.target) + "," + str(j) + ") = " + str(hs))
                            print()
                        similarty_values.append(hs)
                        if hs > thresh:
                            similar_users.append(j)
            except Exception:
                pass
        return similar_users

    def predict(self, similar_users, item):
        r = 0
        c = 0
        for j in similar_users:
            rate = self.R[item - min(self.items)][j - 1]
            if rate != 0:
                r += self.R[item - min(self.items)][j - 1]
                c += 1
        if c != 0:
            return r / c
        else:
            return 0

    def prediction_matrix(self):
        P = np.zeros((len(self.items), len(self.users)))
        for u in self.users:
            print("TARGET: " + str(u))
            rs.set_target_user(u)
            similars = rs.compute_similar_users(0.8, 0.56, verbose=False)
            for i in rs.items:
                pred = rs.predict(similars, i)
                P[i - min(self.items)][u - 1] = pred
        return P

    def search_best_params(self, test_users, thresholds, betas):
        vectors_dict = {}
        for u in test_users:
            self.set_target_user(u)
            vectors_dict[u] = {}
            for t in thresholds:
                vectors_dict[u][t] = {}
                for b in betas:
                    print("user: " + str(u))
                    print("thresh: " + str(t))
                    print("beta: " + str(b))
                    vectors_dict[u][t][b] = {}
                    items = []
                    shared_count = []
                    similars = self.compute_similar_users(t, b, verbose=False)
                    for s in similars:
                        items.append(self.commons_items(self.target, s))
                    all_items = list(itertools.chain.from_iterable(items))
                    all_items.sort()
                    all_items = set(all_items)
                    for it in all_items:
                        c = 0
                        for v in items:
                            if it in v:
                                c += 1
                        shared_count.append(c)
                    vectors_dict[u][t][b]["data"] = shared_count
                    vectors_dict[u][t][b]["total"] = len(items)
            top_sim = 0
            top_tresh = 0
            top_beta = 0
            for u in test_users:
                for t in thresholds:
                    for b in betas:
                        if vectors_dict[u][t][b]["total"] > 5:
                            top = [vectors_dict[u][t][b]["total"] for x in range(0, len(vectors_dict[u][t][b]["data"]))]
                            sim = (1 - cos_dist(top, vectors_dict[u][t][b]["data"])) * vectors_dict[u][t][b]["total"]
                            if sim > top_sim:
                                top_sim = sim
                                top_tresh = t
                                top_beta = b
            with open('result.txt', 'a') as f:
                f.write("TOP PARAMS FOR " + str(u) + " with sim = " + str(top_sim) + " tresh = " + str(
                    top_tresh) + ", beta = " + str(top_beta))
            print("TOP PARAMS FOR " + str(u) + " with sim = " + str(top_sim) + " tresh = " + str(
                top_tresh) + ", beta = " + str(top_beta))


if __name__ == "__main__":
    B_graph = generate_bipartite_graph("rel.rating")
    test_nodes = []
    for i in range(1, 943):
        if B_graph.degree(i) < 40:
            test_nodes.append(i)
    print(i)
    t = np.arange(0.70, 0.90, 0.02).tolist()
    print(t)
    b = np.arange(0.44, 0.66, 0.02).tolist()
    print(b)
    rs = RS(B_graph)
    rs.search_best_params(test_nodes, t, b)
