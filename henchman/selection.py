# -*- coding: utf-8 -*-

'''The selection module.

Contents:
        RandomSelect (df, n_feats): Choose n_feats at random
        Dendrogram (df, pairing_func, max_threshes)
'''
import numpy as np
import pandas as pd
import random

from tqdm import tqdm
from collections import defaultdict


class RandomSelect:
    def __init__(self, names=None, n_feats=0):
        self.names = []
        self.n_feats = n_feats

    def set_params(self, **params):
        # Set parameters of smartselect to **params
        for key in params:
            setattr(self, key, params[key])
        return self

    def fit(self, X, x=None):
        X = pd.DataFrame(X)
        column_list = [i for i in range(0, len(X.columns))]

        random.shuffle(column_list)
        column_list = column_list[:self.n_feats]
        all_columns = X.columns.values
        self.names = [all_columns[item] for item in column_list]

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.names]


class Dendrogram():
    """ Takes in a dataframe and a pairing function.
        Creates a dendrogram which is a set of graphs
        representing connectivity at a set of discrete thresholds.
    """

    def __init__(self, df=None, pairing_func=None, max_threshes=None):
        if pairing_func is None:
            pairing_func = _one_minus_corr
        # Create adjacency matrix and columns list
        self.adj, self.columns = adj_maker(df, pairing_func)

        # Normalize adj
        # self.adj *= np.abs(1.0 / np.abs(self.adj).max())

        # Make edges for every thresh
        self._build_edges(max_threshes)

        # Make graphs for every thresh
        self._build_graphs()

    def _find_all_graphs(self):

        self.graphs = []
        for edges in tqdm(self.edges):
            out = find_connected_components(self.columns.keys(), edges)
            self.graphs.append(out)

    def _build_graphs(self):
        if self.graphs == []:
            self._find_all_graphs()
        for i, name in enumerate(self.graphs):
            if len(name) == 1:
                break
        self.threshlist = self.threshlist[:i]
        self.edges = self.edges[:i]
        self.graphs = self.graphs[:i]

    def _build_edges(self, max_threshes):
        self.edges = []
        self.graphs = []
        uniques = list(np.unique(self.adj))
        # uniques.reverse()
        if max_threshes is None and len(uniques) > 500:
            print('Calculating more than 500 graphs')
            print('You can pass max_threshes as a kwarg to Dendrogram')
        if max_threshes is not None:
            while len(uniques) > max_threshes:
                uniques = uniques[::2]

        self.threshlist = uniques

        for thresh in tqdm(self.threshlist):
            self.edges.append(_edge_maker(self.adj, thresh))

    def features_at_step(self, step):
        '''
        Find the representatives at a certain step for a given graph.
        '''
        featurelist = [self.columns[x] for x in self.graphs[step].iterkeys()]
        return featurelist

    def score_at_point(self, df, labels, step, scoring_func):
        featurelist = self.features_at_step(step)
        print('Using {} features'.format(len(featurelist)))
        return scoring_func(df[featurelist], labels)

    def shuffle_and_score_at_point(self, df, labels, step, scoring_func):
        self._shuffle_all_representatives()
        return self.score_at_point(df, labels, step, scoring_func)

    def find_set_of_size(self, size):
        for i, graph in enumerate(self.graphs):
            if len(graph.keys()) <= size:
                print("There are {} distinct connected components at thresh step {} in the Dendrogram".format(
                    len(graph.keys()), i))
                if i > 0:
                    prevlength = len(self.graphs[i - 1].keys())
                    print("You might also be interested in"
                          " {} components at step {}".format(prevlength, i - 1))
                return i
        print("Warning, could not find requested size, returning last graph")
        return (len(self.graphs) - 1)

    def _shuffle_all_representatives(self):
        assert self.graphs != [], 'Run D._build_graphs to get a graph'
        templist = []
        for graph in self.graphs:
            temp = defaultdict(set)
            for key in graph.keys():
                newkey = random.choice(list(graph[key]))
                temp[newkey] = graph[key]
            templist.append(temp)
        self.graphs = templist

    def transform(self, df, n_feats=10):
        assert df.shape[1] >= n_feats
        step = self.find_set_of_size(n_feats)
        return df[self.features_at_step(step)]


def adj_maker(df, pairing_func):
    '''
    Given a dataframe and a pairing function make
    an adjacency graph and a dictionary of columns.
    The dictionary can be used to associate column position
    to column name.
    '''
    adj = np.zeros((df.shape[1], df.shape[1]))
    for i, col1 in enumerate(df):
        for j, col2 in enumerate(df):
            adj[j][i] = pairing_func(df[col1], df[col2])

    columns = {i: col for i, col in enumerate(df)}
    return adj, columns


def _edge_maker(adj, thresh):
    '''
    Make all edges at a given threshold. Prerequisite
    to make the associated graph.
    '''
    it = np.nditer(adj, flags=['multi_index'])
    edges = []
    for val in it:
        if val <= thresh:
            edges.append(it.multi_index)
    return edges


def find_connected_components(vertices, edges):
    '''
    For vertices and edges (which is a list of 2-tuples),
    do a depth first search to make a dictionary whose
    keys are representatives and values is a list of
    vertices in a given component.
    '''
    d = defaultdict(set)
    for edge in edges:
        d[edge[0]].add(edge[1])
        d[edge[1]].add(edge[0])
    out = defaultdict(set)
    visited_list = []
    temp_list = []
    for vertex in vertices:
        if vertex not in visited_list:
            temp_list.extend([x for x in d[vertex]])
            visited_list.append(vertex)
            out[vertex].add(vertex)
            while temp_list != []:
                newv = temp_list.pop()
                if newv in visited_list:
                    pass
                else:
                    temp_list.extend([x for x in d[newv]])
                    out[vertex].add(newv)
                    visited_list.append(newv)
    return out


def _one_minus_corr(a, b):
    "returns the absolute value of the correlation between a and b"
    return 1 - np.abs(np.corrcoef(a, b, rowvar=False)[0][1])
