#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `learning` module"""
import pandas as pd
import pytest

import henchman.learning as learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score


@pytest.fixture
def Xy():
    X = pd.read_csv('./tests/sample_data/sample_fm_enc.csv')
    y = X.pop('label')
    return X, y


@pytest.fixture
def fm():
    X = pd.read_csv('./tests/sample_data/sample_fm.csv')
    return X


def test_create_holdout(Xy):
    X, y = Xy
    X, X_ho, y, y_ho = learning.create_holdout(X, y)
    assert X.shape[0] == y.shape[0]
    assert X_ho.shape[0] == y_ho.shape[0]
    assert X.shape[1] == X_ho.shape[1]


def test_create_model(Xy):
    X, y = Xy
    score1, _ = learning.create_model(X.iloc[:, :3], y, RandomForestClassifier(), f1_score)
    score2, _ = learning.create_model(X.iloc[:, :3], y, RandomForestClassifier(),
                                      roc_auc_score, n_splits=3)
    assert len(score1) == 1
    assert len(score2) == 3


def test_return_df_shape(Xy):
    X, y = Xy
    out1 = learning.create_model(X.iloc[:, :3], y, RandomForestClassifier(),
                                 f1_score, _return_df=True)
    out2 = learning.create_model(X.iloc[:, :3], y, RandomForestClassifier(),
                                 roc_auc_score, n_splits=3, _return_df=True)
    # Assert same output shape
    assert len(out1) == len(out2)
    assert len(out1[0]) == len(out2[0])
    assert len(out1[1]) == len(out2[1])


def test_inplace_encoder(fm):
    X = fm
    X_new = learning.inplace_encoder(X)
    for col in X_new:
        assert X_new[col].dtype != 'O'
    assert X_new.shape == X.shape


def test_feature_importances(Xy, capsys):
    X, y = Xy
    score, fit_model = learning.create_model(X.iloc[:, :3], y,
                                             RandomForestClassifier(), f1_score)
    out = learning.feature_importances(X, fit_model)
    printed, _ = capsys.readouterr()

    assert len(printed.split('\n')) == 6
    assert len(out) == 3
