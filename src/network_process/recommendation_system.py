import networkx as nx
from util import generate_bipartite_graph
from networkx.algorithms import bipartite
from collections import Counter
from math import log2


class RS:
    def __init__(self, G):
        self.G = G
        self.users, self.items = bipartite.sets(self.G)
        self.R = bipartite.matrix.biadjacency_matrix(self.G, self.users).toarray().tolist()
        self.R_abs_i = None
        self.target = None

    def commons_items(self, node_i, node_j):
        return [x for x in nx.common_neighbors(self.G, node_i, node_j)]

    def set_target_user(self, user):
        self.target = user
        self.abs_difference_rating_matr(self.target)

    def abs_difference_rating_matr(self, user_i):
        self.R_abs_i = {u: {} for u in self.users if u != user_i}
        for user_j in self.users:
            if user_j != user_i:
                commons = self.commons_items(user_i, user_j)
                dict = {}
                for c in commons:
                    dict[c] = abs(self.R[user_i - 1][c - 944] - self.R[user_j - 1][c - 944])
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
        rates = [x for x in self.R[user_id - 1] if x != 0]
        return sum(rates) / len(rates)

    def de(self, user_i, user_j, item):
        return abs(
            (self.R[user_i - 1][item - 944] - self.avg_rate(user_i)) - (
                    self.R[user_j - 1][item - 944] - self.avg_rate(user_j)))

    def deviation(self, user_i, user_j, I):
        commons = self.commons_items(user_i, user_j)
        s = 0
        for i in range(0, I):
            s += self.de(user_i, user_j, commons[i])
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
        c = len(self.commons_items(user_i, user_j))
        m = (self.G.degree(user_i) + self.G.degree(user_j)) / 2
        return c / m

    def hybrid_similarity(self, user_i, user_j, I, beta):
        return (beta * self.simC(user_i, user_j)) + ((1 - beta) * self.simE(user_i, user_j, I))

    def compute_similar_users(self, thresh, verbose=True):
        Wsum = self.weighted_sum(self.target)
        beta = 0.56
        similar_users = []
        similarty_values = []
        for j in self.users:
            if j != self.target:
                I = int(Wsum[j])
                if I > 1:
                    hs = self.hybrid_similarity(self.target, j, I, beta)
                    if verbose:
                        print("Similarity(" + str(self.target) + "," + str(j) + ") = " + str(hs))
                    similarty_values.append(hs)
                    if hs > thresh:
                        similar_users.append(j)
        return similar_users


if __name__ == "__main__":
    B_graph = generate_bipartite_graph("rel.rating")
    rs = RS(B_graph)
    rs.set_target_user(259)
    print(rs.compute_similar_users(0.3))
