#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `learning` module"""
import numpy as np
import pandas as pd
import pytest

import henchman.selection as selection

from henchman.plotting import dendrogram, show
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@pytest.fixture(scope="module")
def Xy():
    X = pd.read_csv('./tests/sample_data/sample_fm_enc.csv')
    y = X.pop('label')
    return X, y


@pytest.fixture(scope="module")
def fit_dend(Xy):
    X, y = Xy
    selector = selection.Dendrogram(X, max_threshes=50)
    return selector


def test_randomselect(Xy):
    X, y = Xy
    selector = selection.RandomSelect()
    selector.set_params(n_feats=10)

    selector.fit(X, y)
    feats = selector.transform(X)
    assert selector.n_feats == 10
    assert feats.shape[1] == 10


def test_dend_fit(fit_dend):
    selector = fit_dend
    assert selector.adj is not None
    assert selector.columns is not None
    assert selector.edges is not None
    assert selector.graphs is not None


def test_dend_set_params(fit_dend):
    threshlist = fit_dend.threshlist
    fit_dend.set_params(threshlist=None)

    assert fit_dend.threshlist is None
    fit_dend.threshlist = threshlist


def test_dend_features_at_step(fit_dend):
    assert len(fit_dend.features_at_step(48)) == 79


def test_dend_find_set_of_size(fit_dend, capsys):
    assert fit_dend.find_set_of_size(80) == 6


def test_dend_score_at_point(Xy, fit_dend):
    X, y = Xy
    scores, fit_model = fit_dend.score_at_point(X, y,
                                                RandomForestClassifier(random_state=0),
                                                accuracy_score, 2)
    assert len(scores) == 1
    assert scores[0] - .866666 < .00001


def test_dend_shuffle_all(fit_dend):
    keys_1 = set(fit_dend.graphs[1].keys())
    fit_dend.shuffle_all_representatives()
    keys_2 = set(fit_dend.graphs[1].keys())
    assert keys_1 != keys_2


def test_dend_shuffle_score_at_point(Xy, fit_dend):
    X, y = Xy
    keys_1 = set(fit_dend.graphs[1].keys())
    scores, _ = fit_dend.shuffle_score_at_point(X, y, RandomForestClassifier(),
                                                accuracy_score, 2, 2)
    assert set(fit_dend.graphs[1].keys()) != keys_1
    assert len(scores) == 2


def test_dend_transform(Xy, fit_dend, capsys):
    X, y = Xy
    X_new_1 = fit_dend.transform(X, 99)
    out1, _ = capsys.readouterr()
    X_new_2 = fit_dend.transform(X, 50)
    out2, _ = capsys.readouterr()

    assert X_new_1.shape[1] == int(out1[10:12])
    assert X_new_2.shape[1] == int(out2[-3:-1])


def test_dend_plot(fit_dend):
    show(dendrogram(fit_dend), static=True)
    show(dendrogram(fit_dend))


def test_build_edges(capsys):
    fake_sel = selection.Dendrogram()
    fake_sel.adj = np.asarray(range(501))
    fake_sel._build_edges(None)

    output, _ = capsys.readouterr()
    split_output = output.split('\n')

    real_line_1 = 'Calculating more than 500 graphs'
    real_line_2 = 'You can pass max_threshes as a kwarg to Dendrogram'
    assert split_output[0] == real_line_1
    assert split_output[1] == real_line_2


def test_build_graphs_exit():
    fake_sel = selection.Dendrogram()
    fake_sel.threshlist = [1, 2]
    fake_sel.edges = [[(0, 1)], [(1, 2), (0, 1)]]
    fake_sel.graphs = [{0: {0, 1}, 2: {2}}, {0: {0, 1, 2}}]
    fake_sel._build_graphs()

    assert fake_sel.threshlist == [1]
    assert fake_sel.edges[0][0][0] == 0
    assert fake_sel.graphs[0][2] == {2}
