import igraph
import numpy as np
import json
from os import path
from util import parseData
import networkx as nx
from teneto import TemporalNetwork as tnTeneto
import progressbar
import chart_studio.plotly as py
import plotly.graph_objects as go


def raws_to_tuple(raws):
    tuples=[]
    for r in raws:
        tuples.append((int(r.split()[0]), int(r.split()[1]), int(r.split()[2]), int(r.split()[3])))
    return tuples


class BipartiteTemporalNetwork:

    def __init__(self, json_file=None, raws_file=None):
        if json_file is not None:
            if path.exists("../../sources/" + json_file):
                self.load_network_from_json("../../sources/" + json_file)
            else:
                print("[NETWORK INIT] sources/" + json_file + " doesn't exist. Generating from source file...")
                if raws_file is None:
                    print("[ERROR] Specify source file since .json file doesn't exist")
                else:
                    print("[NETWORK INIT] Read network from sources/" + raws_file)
                    raws = parseData("../../sources/" + raws_file)
                    raws = raws_to_tuple(raws)
                    self.init_from_raws(raws)
                    self.save_network_to_json("../../sources/" + json_file)
        else:
            self.temporal_edges = []
            self.users = []
            self.movies= []
            self.edges = []
            self.timestamps = []

    def load_from_temporal_edges(self, temporal_edges, timestamps=None):
        print("[NETWORK INIT] Read temporal edges...")
        self.temporal_edges = temporal_edges
        print("[NETWORK INIT] Generate nodes...")
        self.users = self.get_users_and_movies()[0]
        self.movies = self.get_users_and_movies()[1]
        print("[NETWORK INIT] Generate timestamps...")
        if timestamps:
            self.timestamps = timestamps
        else:
            self.timestamps = [x for x in range(0, len(self.temporal_edges))]
        print("[NETWORK INIT] Generate aggregated edges...")
        self.edges = self.get_edges()
        print("*** TEMPORAL NETWORK INITIALIZED ***")
        print()

    def init_from_raws(self, raws, span=943):
        # span is use to remap movies ids for having different nodes id for users and movies
        edges_list = []
        timestamp_list = []
        for link in raws:
            node_in = int(link[0])
            node_out = span + int(link[1])
            weight = int(link[2])
            timestamp = int(link[3])
            timestamp_list.append(timestamp)
            edges_list.append((node_in, node_out, weight, timestamp))
        timestamps = np.unique(timestamp_list).tolist()
        self.timestamps = timestamps
        temporal_edges = []
        bar = progressbar.ProgressBar()
        for i in bar(range(len(timestamps))):
            l = [[y[0], y[1], y[2]] for y in edges_list if y[3] == timestamps[i]]
            temporal_edges.insert(timestamps[i], l)
        self.temporal_edges = temporal_edges
        print("[NETWORK INIT] Generate nodes...")
        self.users = self.get_users_and_movies()[0]
        self.movies = self.get_users_and_movies()[1]
        print("[NETWORK INIT] Generate aggregated edges...")
        self.edges = self.get_edges()

    def get_users_and_movies(self):
        users = []
        movies = []
        for edges in self.temporal_edges:
            for link in edges:
                users.append(link[0])
                movies.append(link[1])
        return set(users), set(movies)

    def get_edges(self):
        e = []
        for t, edges in enumerate(self.temporal_edges):
            ts = self.timestamps[t]
            for link in edges:
                e.append((link[0], link[1], ts, link[2]))
        y = np.unique(e, axis=0)
        z = []
        for i in y:
            z.append(tuple(i))
        return z

    def load_network_from_json(self, file_name):
        with open(file_name) as json_file:
            data = json.load(json_file)
        self.load_from_temporal_edges(data["edges"], data["timestamps"])

    def save_network_to_json(self, file_name):
        graph = {}
        graph["edges"] = self.temporal_edges
        graph["timestamps"] = self.timestamps
        with open(file_name, 'w') as outfile:
            json.dump(graph, outfile)

    def to_teneto(self):
        ttn = tnTeneto()
        print(self.edges)
        ttn.network_from_edgelist(self.edges)
        return ttn

    def to_aggregate_networkX(self):
        return

