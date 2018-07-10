# -*- coding: utf-8 -*-

'''The learning module. Do machine learning.

Contents:
        create_model: Makes a model.
        inplace_encoder: Label encodes all columns with dtype = 'O'.
        feature_importances: Prints most important features in a model.

'''
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder


def _fit_predict(X_train, X_test, y_train, y_test, model, metric):
    model = model
    model.fit(X_train, y_train)
    if metric.func_name == 'roc_auc_score':
        return metric(y_test, model.predict_proba(X_test)[:, 1]), model
    preds = model.predict(X_test)
    return metric(y_test, preds), model


def _score_tt(X, y, model, metric):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    score, fit_model = _fit_predict(X_train, X_test,
                                    y_train, y_test, model, metric)
    return [score], fit_model


def create_model(X, y, model=None, metric=None, n_splits=1):
    '''Make a model from X and y. Returns a scorelist and a fit model.
    Input:
        X (df): A cleaned numeric feature matrix
        y (series): A column of labels
        model: A sklearn model with fit and predict methods
        metric: A metric which takes y_test, preds and returns a score
        n_splits: If 1 use a train_test_split. Otherwise use tssplit.

    '''
    assert np.array_equal(X.index, y.index)
    assert model is not None
    assert metric is not None
    if n_splits == 1:
        return _score_tt(X, y, model, metric)

    if n_splits > 1:
        scorelist = []
        tssplit = TimeSeriesSplit(n_splits=n_splits)
        for i, (train_index, test_index) in enumerate(tssplit.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            score, fit_model = _fit_predict(X_train, X_test,
                                            y_train, y_test, model, metric)
            scorelist.append(score)
        return scorelist, fit_model


def inplace_encoder(X):
    '''Replace all columns with pd.dtype == 'O' with integers.
    This avoids the dimensionality problems of OHE at the cost of
    implying an artificial ordering in categorical features.

    Input:
        X (df): The dataframe to encode
    Output:
        X (df): An encoded dataframe
    '''
    for col in X:
        if X[col].dtype == 'O':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[[col]].astype(str))
    return X


def _raw_feature_importances(X, model):
    feature_imps = [(imp, X.columns[i])
                    for i, imp in enumerate(model.feature_importances_)]
    feature_imps.sort()
    feature_imps.reverse()
    return feature_imps


def feature_importances(X, model, n_feats=5):
    feature_imps = _raw_feature_importances(X, model)
    for i, f in enumerate(feature_imps[0:n_feats]):
        print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]/feature_imps[0][0]))
    print('-----\n')
    return [f[1] for f in feature_imps[:n_feats]]
