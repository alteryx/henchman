#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `learning` module"""
import pandas as pd
import pytest

import henchman.learning as learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


@pytest.fixture
def df():
    """See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    X = pd.read_csv('./docs/fm.csv')
    y = X.pop('label')
    return X, y


def test_create_holdout(df):
    X, y = df
    X, X_ho, y, y_ho = learning.create_holdout(X, y)
    assert X.shape[0] == y.shape[0]
    assert X_ho.shape[0] == y_ho.shape[0]
    assert X.shape[1] == X_ho.shape[1]


def test_create_model(df):
    X, y = df
    score1, _ = learning.create_model(X.iloc[:, :3], y, RandomForestClassifier(), f1_score)
    score2, _ = learning.create_model(X.iloc[:, :3], y, RandomForestClassifier(),
                                      f1_score, n_splits=3)
    assert len(score1) == 1
    assert len(score2) == 3


def test_inplace_encoder(df):
    X, y = df
    X_new = learning.inplace_encoder(X)
    for col in X_new:
        assert X_new[col].dtype != 'O'
    assert X_new.shape == X.shape


def test_feature_importances(df, capsys):
    X, y = df
    score, fit_model = learning.create_model(X.iloc[:, :3], y,
                                             RandomForestClassifier(), f1_score)
    out = learning.feature_importances(X, fit_model)
    printed, _ = capsys.readouterr()

    assert len(printed.split('\n')) == 6
    assert len(out) == 3
