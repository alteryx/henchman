# -*- coding: utf-8 -*-

'''The selection module.

Contents:
        RandomSelect (X, n_feats): Choose n_feats at random
        Dendrogram (X, pairing_func, max_threshes)
'''
import numpy as np
import pandas as pd
import math
import random

from tqdm import tqdm
from collections import defaultdict

from henchman.learning import create_model


class RandomSelect:
    """Randomly choose a feature set.
    """

    def __init__(self, names=None, n_feats=0):
        '''A class for randomly choosing a feature set.

        Args:
            names (list[str]): A list of column names selected. Default is the empty list.
            n_feats (int): The number of features to randomly select.
        '''
        self.names = []
        self.n_feats = n_feats

    def set_params(self, **params):
        '''Method to functionally assign parameters.
        Expects a dictionary ``**params`` as input.
        '''
        for key in params:
            setattr(self, key, params[key])
        return self

    def fit(self, X, x=None):
        '''Randomly choose which features to select.
        Args:
            X (pd.Dataframe): A dataframe from which to select
                a subset of columns.
        '''
        X = pd.DataFrame(X)
        column_list = [i for i in range(0, len(X.columns))]

        random.shuffle(column_list)
        column_list = column_list[:self.n_feats]
        all_columns = X.columns.values
        self.names = [all_columns[item] for item in column_list]

    def transform(self, X):
        '''Returns a subset of a dataframe.
        Args:
            X (pd.DataFrame): A dataframe with the same
                column names as the one with which the selector
                was fit.
        Returns:
            X_trans (pd.DataFrame): The dataframe subset X[self.names].
        '''
        X = pd.DataFrame(X)
        return X[self.names]


class Dendrogram():
    """ Pair features by an arbitrary function.
        Creates a dendrogram which is a set of graphs
        representing connectivity at a set of discrete thresholds.
    """

    def __init__(self, X=None, pairing_func=None, max_threshes=None):
        '''An object to store graphs for a given pairing function.
        If given a dataframe X this first creates an
        adjacency matrix given a certain pairing function.
        It will then go through and build endges and graphs
        from those edge-vertex pairs. The graphs are all
        stored in order.

        Args:
            X (pd.DataFrame): The dataframe for which to build the Dendrogram.
            pairing_func (func): A function which takes in two columns and
                returns a number.
            max_threshes (int): The maximum number of graphs to build.

        '''
        if X is not None:
            self.fit(X, pairing_func=pairing_func, max_threshes=max_threshes)

    def fit(self, X, pairing_func=None, max_threshes=None):
        '''Build graphs for a given pairing function.
        First creates an adjacency matrix given a certain pairing function.
        It will then go through and build endges and graphs from those
        edge-vertex pairs. The graphs are all stored in order.

        Args:
            X (pd.DataFrame): The dataframe for which to build the Dendrogram.
            pairing_func (func): A function which takes in two columns and
                returns a number.
            max_threshes (int): The maximum number of graphs to build.
        '''
        if pairing_func is None:
            pairing_func = _one_minus_corr

        # Create adjacency matrix and columns list
        self.adj, self.columns = adj_maker(X, pairing_func)

        # Make edges for every thresh
        self._build_edges(max_threshes)

        # Make graphs for every thresh
        self._build_graphs()

        assert len(self.edges) > 0, 'Failed to build edges'
        assert len(self.graphs) > 0, 'Failed to build graphs'

    def set_params(self, **params):
        '''Method to functionally assign parameters.
        Expects a dictionary ``**params`` as input.
        '''
        for key in params:
            setattr(self, key, params[key])
        return self

    def _find_all_graphs(self):

        self.graphs = []
        for edges in tqdm(self.edges):
            out = find_connected_components(list(self.columns.keys()), edges)
            self.graphs.append(out)

    def _build_graphs(self):
        if self.graphs == []:
            self._find_all_graphs()
        for i, graph in enumerate(self.graphs):
            if len(graph) == 1:
                break
        self.threshlist = self.threshlist[:i]
        self.edges = self.edges[:i]
        self.graphs = self.graphs[:i]

    def _build_edges(self, max_threshes):
        self.edges = []
        self.graphs = []
        uniques = list(np.unique(self.adj))
        uniques = [x for x in uniques if not math.isnan(x)]
        if max_threshes is None and len(uniques) > 500:
            print('Calculating more than 500 graphs')
            print('You can pass max_threshes as a kwarg to Dendrogram')
        if max_threshes is not None:
            if len(uniques) > max_threshes:
                # Take max_threshes evenly spaced thresholds
                uniques = uniques[::int(np.floor((len(uniques)/max_threshes)) + 1)]

        self.threshlist = uniques

        for thresh in tqdm(self.threshlist):
            self.edges.append(_edge_maker(self.adj, thresh))

    def features_at_step(self, step):
        '''Find the representatives at a certain step for a given graph.

        Args:
            step (int): Which position in self.threshlist to show features from.

        Returns:
            A list of features at ``step``. (list[str])
        '''
        featurelist = [self.columns[x] for x, _ in self.graphs[step].items()]
        return featurelist

    def score_at_point(self, X, y, model, metric, step, n_splits=1):
        '''A helper method for scoring a Dendrogram at a step.

        Args:
            X (pd.DataFrame): A dataframe with the same columns that the
                Dendrogram was built with.
            y (pd.Series): Labels for X.
            step (int): Which position in self.threshlist to show features from.
            scoring_func (func): A function which takes in X and y.
        '''
        featurelist = self.features_at_step(step)
        print('Using {} features'.format(len(featurelist)))
        return create_model(X[featurelist], y, model, metric, n_splits=n_splits)

    def shuffle_score_at_point(self, X, y, model, metric, step, n_splits=1):
        '''A helper method for scoring a Dendrogram at a step.
        This method shuffles the graph representatives and then
        runs ``score_at_point``. By running shuffle and score at point
        multiple times, you can get an impression of how representative
        a particular feature set is of the underlying graph structure.

        Args:
            X (pd.DataFrame): A dataframe with the same columns that the
                Dendrogram was built with.
            y (pd.Series): Labels for X.
            model: A sklearn model with fit and predict methods.
            metric: A metric which takes y_test, preds and returns a score.
            step (int): Which position in self.threshlist to show features from.
            n_splits (int): If 1 use a train_test_split. Otherwise use tssplit.
                        Default value is 1.
        '''

        self.shuffle_all_representatives()
        return self.score_at_point(X, y, model, metric, step, n_splits=n_splits)

    def find_set_of_size(self, size):
        '''Finds a column set of a certain size in the Dendrogram.
        This checks graphs until there are only ``size`` remaining components.

        Args:
            size (int): The number of features you want to end up with.

        Returns:
            The step at which there are ``size`` connected components. (int)
        '''
        for i, graph in enumerate(self.graphs):
            if len(graph.keys()) <= size:
                print("There are {} distinct connected components "
                      "at thresh step {} in the Dendrogram".format(
                          len(graph.keys()), i))
                if i > 0:
                    prevlength = len(self.graphs[i - 1].keys())
                    print("You might also be interested in"
                          " {} components at step {}".format(prevlength, i - 1))
                return i
        print("Warning, could not find requested size, returning set of size {}".format(
            len(self.graphs[-1])))
        return (len(self.graphs) - 1)

    def shuffle_all_representatives(self):
        '''Shuffle representatives for every graph in ``D.graphs``.
        For every graph, look through the list associated to
        each key and choose one to be the new key. Note that keys
        are not fixed step by step, so the key for a cluster at step n
        is not guarenteed to be the key for the same cluster at step n+1.
        '''
        assert self.graphs != [], 'Run D._build_graphs to get a graph'
        templist = []
        for graph in self.graphs:
            temp = defaultdict(set)
            for key in graph.keys():
                newkey = random.choice(list(graph[key]))
                temp[newkey] = graph[key]
            templist.append(temp)
        self.graphs = templist

    def transform(self, X, n_feats=10):
        '''Return a dataframe of a particular size.

        Args:
            X (pd.Dataframe): The dataframe to transform.
            n_feats (int): The number of columns to return
        '''
        assert X.shape[1] >= n_feats
        step = self.find_set_of_size(n_feats)
        return X[self.features_at_step(step)]


def adj_maker(data, pairing_func):
    '''Given a dataframe and a pairing function make
    an adjacency graph and a dictionary of columns.
    The dictionary can be used to associate column position
    to column name.

    Args:
        data (pd.DataFrame): A dataframe from which to make an
            adjacency graph.
        pairing_function (func): A function which takes in two columns
            and returns a number.

    Returns:
        adj, columns (np.array, dict[int, str]): An adjacency graph
            and a dictionary pairing column locations with column names.
    '''
    adj = np.zeros((data.shape[1], data.shape[1]))
    for i, col1 in enumerate(data):
        for j, col2 in enumerate(data):
            adj[j][i] = pairing_func(data[col1], data[col2])

    columns = {i: col for i, col in enumerate(data)}
    return adj, columns


def _edge_maker(adj, thresh):
    '''Make all edges at a given threshold. Prerequisite
    to make the associated graph.
    '''
    it = np.nditer(adj, flags=['multi_index'])
    edges = []
    for val in it:
        if val <= thresh:
            edges.append(it.multi_index)
    return edges


def find_connected_components(vertices, edges):
    '''Make a graph from a list of vertices and edges.
    Do a depth first search to make a dictionary whose
    keys are representatives and values is a list of
    vertices in a given component. It is assumed all edges
    are symmetric.

    Args:
        vertices (list[int]): A list of vertex locations.
        edges (list[tuple[int, int]]): A list of pairs of vertices.
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
