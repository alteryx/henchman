#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `learning` module"""
import pandas as pd
import pytest

import henchman.plotting as hplot
from henchman.learning import create_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


@pytest.fixture
def fm():
    fm = pd.read_csv('./tests/sample_data/sample_fm.csv')
    return fm


@pytest.fixture
def Xy():
    X = pd.read_csv('./tests/sample_data/sample_fm_enc.csv')
    y = X.pop('label')
    return X, y


def test_show_template(capsys):
    hplot.show_template()
    output_str, _ = capsys.readouterr()
    output = output_str.split('\n')
    trueoutput = ['show(plot,',
                  '     static=False,',
                  '     png=False,',
                  '     hover=False,',
                  '     colors=None,',
                  '     width=None,',
                  '     height=None,',
                  "     title='Temporary title',",
                  "     x_axis='my xaxis name',",
                  "     y_axis='my yaxis name',",
                  '     x_range=(0, 10) or None,',
                  '     y_range=(0, 10) or None)', '', '']

    for i, value in enumerate(trueoutput):
        assert output[i] == value


def test_make_piechart_source(fm):
    pie_df = hplot._make_piechart_source(fm['flights.carrier'], mergepast=2, sort=False, drop_n=1,
                                         figargs={'colors': None})
    names = pie_df['names'].values
    truevalues = ['AS', 'B6', 'Other']

    assert pie_df.shape[1] == 6
    for i, value in enumerate(truevalues):
        assert names[i] == value


def test_make_histogram_source(fm):
    hist_df1 = hplot._make_histogram_source(fm['flights.distance_group'],
                                            y=None,
                                            n_bins=10,
                                            col_max=13,
                                            col_min=-1,
                                            normalized=True)
    hist_df2 = hplot._make_histogram_source(fm['flights.distance_group'],
                                            y=fm['label'],
                                            n_bins=10,
                                            col_max=None,
                                            col_min=None,
                                            normalized=False)
    assert hist_df1.shape[1] == (hist_df2.shape[1] - 1)

    truehist = [12, 21, 19, 12, 10, 4, 4, 2, 3, 13]
    for i, value in enumerate(truehist):
        assert hist_df2['hist'].values[i] == value


def test_make_timeseries_source(fm):
    fm_with_time = fm.reset_index()
    fm_with_time['time'] = pd.to_datetime(fm_with_time['time'])
    col_1 = fm_with_time['time']
    col_2 = fm_with_time['label']

    time_df = hplot._make_timeseries_source(col_1, col_2)
    truelabelmean = [0.11111111, 0.09090909, 0., 0.18181818, 0.11111111,
                     0.08333333,  0.22222222, 0., 0.2, 0.07692308]
    for i, value in enumerate(truelabelmean):
        assert (time_df['height'].values[i] - value) < .00001


def test_make_scatter_source(fm):

    col_1 = fm['scheduled_elapsed_time']
    col_2 = fm['distance']
    agg = fm['flights.carrier']
    scatter_df = hplot._make_scatter_source(col_1, col_2, agg, label=fm['label'], aggregate='last')

    true_last_distance = [1258., 954., 200., 1535., 2556., 236., 612., 867., 2288., 967.]
    for i, value in enumerate(true_last_distance):
        assert scatter_df['col_2'].values[i] == value


def test_piechart(fm):
    hplot.show(hplot.piechart(fm['flights.carrier']))
    hplot.show(hplot.piechart(fm['flights.carrier']), static=True)


def test_histogram(fm):
    hplot.show(hplot.histogram(fm['flights.distance_group']))
    hplot.show(hplot.histogram(fm['flights.distance_group']), static=True)


def test_timeseries(fm):
    fm_with_time = fm.reset_index()
    fm_with_time['time'] = pd.to_datetime(fm_with_time['time'])
    col_1 = fm_with_time['time']
    col_2 = fm_with_time['label']
    hplot.show(hplot.timeseries(col_1, col_2))
    hplot.show(hplot.timeseries(col_1, col_2), static=True)


def test_scatter(fm):
    col_1 = fm['scheduled_elapsed_time']
    col_2 = fm['distance']
    agg = fm['flights.carrier']
    hplot.show(hplot.scatter(col_1, col_2, agg))
    hplot.show(hplot.scatter(col_1, col_2, agg), static=True)


def test_feature_importances_plot(Xy):
    X, y = Xy
    scores, fit_model = create_model(X, y, RandomForestClassifier(), roc_auc_score)


def test_roc_auc(Xy):
    X, y = Xy
    hplot.show(hplot.roc_auc(X, y, RandomForestClassifier(), n_splits=3))


def test_f1(Xy):
    X, y = Xy
    hplot.show(hplot.f1(X, y, RandomForestClassifier()))


def test_modify_plot(fm):
    hplot.show(hplot.histogram(fm['flights.distance_group']),
               width=300, height=300, title='Distance Histogram',
               x_axis='axis x', y_axis='axis y',
               x_range=(0, 1000), y_range=(0, 1000))


def test_color(fm):
    colors = ['white', '#000000', 'green']
    hplot.show(hplot.histogram(fm['flights.distance_group']), colors=colors)
    hplot.show(hplot.piechart(fm['flights.dest']), colors=colors)
    hplot.show(hplot.scatter(fm['distance'], fm['distance']), colors=colors)

    fm_with_time = fm.reset_index()
    fm_with_time['time'] = pd.to_datetime(fm_with_time['time'])
    col_1 = fm_with_time['time']
    col_2 = fm_with_time['label']
    hplot.show(hplot.timeseries(col_1, col_2), colors=colors)


def test_gridplot(fm):
    p1 = hplot.show(hplot.histogram(fm['flights.distance_group']), fig=True, static=True)
    p2 = hplot.show(hplot.histogram(fm['flights.distance_group'], n_bins=15), fig=True, static=True)
    hplot.gridplot([p1, p2], n_cols=2)
    hplot.gridplot([[p1], [p2]])
