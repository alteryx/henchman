# -*- coding: utf-8 -*-

'''The learning module. Do machine learning.

Contents:
        create_validation: A wrapper around sklearn train_test_split.
        create_model: Makes a model.
        inplace_encoder: Label encodes all columns with dtype = 'O'.
        feature_importances: Prints most important features in a model.

'''
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder


def create_holdout(X, y, split_size=.3):
    '''A wrapper around train_test_split.

    Args:
        X (pd.DataFrame): The dataframe to split.
        y (pd.Series): The labels to split.
        split_size (float): Size of testing set. Default is .3.

    Example:
        >>> from henchman.learning import create_holdout
        >>> X, X_ho, y, y_ho = create_holdout(X, y)
    '''
    return train_test_split(X, y, shuffle=False, test_size=split_size)


def _fit_predict(X_train, X_test, y_train, y_test, model, metric):
    model = model
    model.fit(X_train, y_train)

    if metric.__name__ == 'roc_auc_score':
        return metric(y_test, model.predict_proba(X_test)[:, 1]), model

    preds = model.predict(X_test)
    return metric(y_test, preds), model


def _score_tt(X, y, model, metric, split_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=split_size)
    score, fit_model = _fit_predict(X_train, X_test,
                                    y_train, y_test, model, metric)
    return [score], fit_model


def create_model(X, y, model=None, metric=None,
                 n_splits=1, split_size=.3, _return_df=False):
    '''Make a model. Returns a scorelist and a fit model.
    A wrapper around a standard scoring workflow. Uses
    ``train_test_split`` unless otherwise specified (in which case
    it will use ``TimeSeriesSplit``).

    In this function we trade flexibility for ease of use. Unless
    you want this exact validation-fitting-scoring method, it's
    recommended you just use the sklearn API.

    Args:
        X (pd.DataFrame): A cleaned numeric feature matrix.
        y (pd.Series): A column of labels.
        model: A sklearn model with fit and predict methods.
        metric: A metric which takes y_test, preds and returns a score.
        n_splits (int): If 1 use a train_test_split. Otherwise use tssplit.
                Default value is 1.
        split_size (float): Size of testing set. Default is .3.
        _return_df (bool): If true, return (X_train, X_test, y_train, y_test) after returns.
                Not generally useful, but sometimes necessary.

    Returns:
        (list[float], sklearn.ensemble): A list of scores and a fit model.

    Example:
        >>> from henchman.learning import create_model
        >>> import numpy as np
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.metrics import roc_auc_score
        >>> scores, fit_model = create_model(X, y,
        ...                                  RandomForestClassifier(),
        ...                                  roc_auc_score,
        ...                                  n_splits=5)
        >>> print('Average score of {:.2f}'.format(np.mean(scores)))

    '''
    assert np.array_equal(X.index, y.index)
    assert model is not None
    assert metric is not None
    if n_splits == 1:
        if _return_df:
            return _score_tt(X, y, model, metric, split_size), create_holdout(X, y, split_size)
        return _score_tt(X, y, model, metric, split_size)

    if n_splits > 1:
        scorelist = []
        tssplit = TimeSeriesSplit(n_splits=n_splits)
        for i, (train_index, test_index) in enumerate(tssplit.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            score, fit_model = _fit_predict(X_train, X_test,
                                            y_train, y_test, model, metric)
            scorelist.append(score)
        if _return_df:
            return (scorelist, fit_model), (X_train, X_test, y_train, y_test)
        return scorelist, fit_model


def inplace_encoder(X):
    '''Replace all columns with pd.dtype == 'O' with integers.
    This avoids the dimensionality problems of OHE at the cost of
    implying an artificial ordering in categorical features.

    Args:
        X (pd.DataFrame): The dataframe to encode.

    Returns:
       pd.DataFrame: A dataframe whose categorical columns have been replaced by integers.

    Example:
        >>> from henchman.learning import inplace_encoder
        >>> X_enc = inplace_encoder(X)
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
    '''Print a list of important features.
    Also returns a list of column names.

    Args:
        X(pd.DataFrame): The dataframe from which the features are drawn.
        model(sklearn.ensemble): A model with a ``feature_importances_`` attribute.
        n_feats(int): Number of feature importances to return.

    Returns:
        list[str]: A list of n_feats feature column names.

    Example:
        >>> from henchman.learning import feature_importances
        >>> my_feats = feature_importances(X, fit_model, n_feats=5)
        >>> X[my_feats].head()
    '''
    feature_imps = _raw_feature_importances(X, model)
    for i, f in enumerate(feature_imps[0:n_feats]):
        print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]/feature_imps[0][0]))
    print('-----\n')
    return [f[1] for f in feature_imps[:n_feats]]
