# -*- coding: utf-8 -*-

'''The selection module.

Contents:
        RandomSelect (df, n_feats): Choose n_feats at random
'''
import pandas as pd
import random


class RandomSelect:
    def __init__(self, names=[], n_feats=0):
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
        column_list = column_list[:self.nfeatures]
        all_columns = X.columns.values
        self.names = [all_columns[item] for item in column_list]

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.names]
