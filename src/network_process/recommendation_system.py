import networkx as nx
from util import generate_bipartite_graph
from networkx.algorithms import bipartite
from collections import Counter
from math import log2
import numpy as np
from scipy.spatial.distance import cosine as cos_dist
import progressbar
from scipy.stats import pearsonr


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
        '''
        Init system with bipartite graph
        :param G: bipartite Graph, generated from util.generate_bipartite_graph
        :var users: user ids in G
        :var items: item ids in G
        :var R_abs_i: absolute difference matrix (dictionary) between user i and all others users j.
                      R_abs_i[j][it] abs difference between i and j over common item it
        :var target: id of target user
        '''
        self.G = G
        self.users, self.items = bipartite.sets(self.G)
        self.R = bipartite.matrix.biadjacency_matrix(self.G, self.items).toarray().tolist()
        self.R_abs_i = None
        self.target = None

    def info(self):
        '''
        Print System Information (Matrix dimensions, users_ids, item_ids...)
        :return:
        '''
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
        '''
        :param node_i: user_i id
        :param node_j: user_j id
        :return: list of common rated items between i and j
        '''
        l = [x for x in nx.common_neighbors(self.G, node_i, node_j)]
        l.sort()
        return l

    def set_target_user(self, user):
        '''
        See also abs_difference_rating_matr
        :param user: target user id
        :return:
        '''
        self.target = user
        self.abs_difference_rating_matr(self.target)

    def abs_difference_rating_matr(self, user_i):
        '''
        Fill R_abs_i matrix for user i. Call in target user setter.
        :param user_i: user i id
        :return:
        '''
        self.R_abs_i = {u: {} for u in self.users if u != user_i}
        for user_j in self.users:
            if user_j != user_i:
                commons = self.commons_items(user_i, user_j)
                dict = {}
                for c in commons:
                    dict[c] = abs(self.R[c - min(self.items)][user_i - 1] - self.R[c - min(self.items)][user_j - 1])
                self.R_abs_i[user_j] = dict

    def count_abs_rating_difference(self, user_j):
        '''
        Util function for Equation 2 in paper.
        :param user_j:
        :return: a dictionary that contains for each abs_rate_difference (0 (i.e 5-5) to 4 (i.e 5-1)) the number of
                 items between target and user_j with abs_rate_difference.
        '''
        dict = {}
        for i in range(5):
            count = Counter(self.R_abs_i[user_j].values())[i]
            dict[i] = count
        return dict

    def weighted_sum(self, user_i):
        '''
        :param user_i: user id (target)
        :return: a dictionary with key values all user_j != target and values the weighted sum p(i,j)
                 as defined in Equation 2
        '''
        dict = {}
        for user_j in self.users:
            if user_j != user_i:
                count = self.count_abs_rating_difference(user_j)
                dict[user_j] = 1 * count[0] + 0.8 * count[1] + 0.6 * count[2] + 0.4 * count[3] + 0.2 * count[4]
        return dict

    def avg_rate(self, user_id):
        '''
        :param user_id: user id
        :return: average rate based on user_id rates
        '''
        rates = [x for x in get_column(user_id - 1, self.R) if x != 0]
        return sum(rates) / len(rates)

    def de(self, user_i, user_j, item):
        '''
        :param user_i:
        :param user_j:
        :param item:
        :return: value defined in Equation 5.
        '''
        return abs(
            (self.R[item - min(self.items)][user_i - 1] - self.avg_rate(user_i)) - (
                    self.R[item - min(self.items)][user_j - 1] - self.avg_rate(user_j)))

    def deviation(self, user_i, user_j, I):
        '''
        :param user_i:
        :param user_j:
        :param I: cardinality of filtered common items based on weighted sum
        :return: Deviation across all the co-rated items for the user U i and U j is represented as in Eq. (6)
        '''
        commons = self.commons_items(user_i, user_j)
        s = 0
        for i in range(0, I):
            s += self.de(user_i, user_j, commons[i])
        if s != 0:
            return s

    def probability_function(self, user_i, user_j, item_k, I):
        '''
        :param user_i:
        :param user_j:
        :param item_k:
        :param I:
        :return: p function Equation 7
        '''
        return self.de(user_i, user_j, item_k) / self.deviation(user_i, user_j, I)

    def entropy(self, user_i, user_j, I):
        '''
        :param user_i:
        :param user_j:
        :param I:
        :return: entropy given in Equation 8
        '''
        commons = self.commons_items(user_i, user_j)
        H = 0
        for i in range(0, I):
            pk = self.probability_function(user_i, user_j, commons[i], I)
            H += pk * log2(pk)
        return -H

    def simE(self, user_i, user_j, I):
        '''
        :param user_i:
        :param user_j:
        :param I:
        :return: simE given in Equation 9
        '''
        return 1 - (self.entropy(user_i, user_j, I) / log2(I))

    def simC(self, user_i, user_j):
        '''
        :param user_i:
        :param user_j:
        :return: Cosine similarity
        '''
        num = 0
        sum1 = 0
        sum2 = 0
        for it in self.items:
            num += self.R[it - 1001][user_i - 1] * self.R[it - 1001][user_j - 1]
            sum1 += np.power(self.R[it - 1001][user_i - 1], 2)
            sum2 += np.power(self.R[it - 1001][user_j - 1], 2)
        den = np.sqrt(sum1) * np.sqrt(sum2)
        return num / den

    def hybrid_similarity(self, user_i, user_j, I, beta, verbose):
        '''
        :param user_i:
        :param user_j:
        :param I:
        :param beta:
        :param verbose:
        :return: Hybrid similarity proposed in the paper, Equation 10
        '''
        sim_c = self.simC(user_i, user_j)
        sim_e = self.simE(user_i, user_j, I)
        if verbose:
            print("Cosine: " + str(sim_c))
            print("Entropy: " + str(sim_e))
        return (beta * sim_c) + ((1 - beta) * sim_e)

    def compute_similar_users(self, thresh, beta, nodes=None, verbose=True):
        '''
        :param thresh:
        :param beta:
        :param nodes:
        :param verbose:
        :return: Compute similar users . Similar users have hybrid_similarity > thresh
        '''
        Wsum = self.weighted_sum(self.target)
        similar_users = []
        similarty_values = []
        if nodes is None:
            users_to_compute = self.users
        else:
            users_to_compute = nodes
        if verbose:
            bar = list
        else:
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
        '''
        :param similar_users:
        :param item:
        :return: Predicted rates based on similar users (mean of rates)
        '''
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
        '''
        :return: generate for all users a matrix with all predictions for all items (HUGE COMPUTATION)
        '''
        P = np.zeros((len(self.items), len(self.users)))
        for u in self.users:
            print("TARGET: " + str(u))
            rs.set_target_user(u)
            similars = rs.compute_similar_users(0.8, 0.56, verbose=False)
            for i in rs.items:
                pred = rs.predict(similars, i)
                P[i - min(self.items)][u - 1] = pred
        return P

    def compare_predicted_real(self, th=0.7, b=0.7):
        '''
        :param th:
        :param b:
        :return: Print real rates and predicted for already rated items by target user
        '''
        similars = rs.compute_similar_users(th, b, verbose=False)
        for i in rs.items:
            pred = rs.predict(similars, i)
            real = rs.R[i - 1001][self.target - 1]
            if real != 0:
                print("ITEMS: ", str(i))
                print("PREDICTION:" + str(pred))
                print("REAL:" + str(real))
                print()

    def predict_unrated_item(self, th=0.7, b=0.7):
        '''
        :param th:
        :param b:
        :return: Print real rates and predicted for already UNrated items by target user
        '''
        similars = rs.compute_similar_users(th, b, verbose=False)
        for i in rs.items:
            pred = rs.predict(similars, i)
            real = rs.R[i - 1001][self.target - 1]
            if real == 0:
                print("ITEMS: ", str(i))
                print("PREDICTION:" + str(pred))
                print("REAL:" + str(real))
                print()


if __name__ == "__main__":
    B_graph = generate_bipartite_graph("rel.rating")
    rs = RS(B_graph)
    rs.set_target_user(4)
    similars = rs.compute_similar_users(0.5, 0.40, verbose=False)
    print(similars)
