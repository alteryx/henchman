# -*- coding: utf-8 -*-

'''The plotting module.

Contents:
        feature_importances_plot
'''
import pandas as pd
import numpy as np

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.io import output_notebook

from henchman.learning import _raw_feature_importances


def feature_importances(X, model, n_feats=5):
    '''Plot feature importances.
    Input:
        raw_feature_imps (list[tuple[float, str]]): Complete list of
                feature importances.
    '''
    feature_imps = _raw_feature_importances(X, model)
    features = [f[1] for f in feature_imps[0:n_feats]][::-1]
    importances = [f[0] for f in feature_imps[0:n_feats]][::-1]

    output_notebook()
    source = ColumnDataSource(data={'feature': features,
                                    'importance': importances})
    p = figure(y_range=features,
               height=500,
               title="Random Forest Feature Importances")
    p.hbar(y='feature',
           right='importance',
           height=.8,
           left=0,
           source=source,
           color="#008891")
    p.toolbar_location = None
    p.yaxis.major_label_text_font_size = '10pt'
    return p


def static_histogram(col, n_bins=10,
                     col_max=None,
                     col_min=None):
    hover = HoverTool(
        tooltips=[
            ("Height", " @hist"),
            ("Bin", " [@left{0.00}, @right{0.00})"),
        ],
        mode='mouse')
    if col_max is None:
        col_max = col.max()
    if col_min is None:
        col_min = col.min()
    truncated = col[(col <= col_max) & (col >= col_min)]
    hist, edges = np.histogram(truncated, bins=n_bins, density=False)
    source = ColumnDataSource(pd.DataFrame({'hist': hist,
                                            'left': edges[:-1],
                                            'right': edges[1:]}
                                           ))

    plot = figure(tools=[hover, 'box_zoom', 'save', 'reset'])
    plot.quad(top='hist', bottom=0,
              left='left', right='right',
              line_color='white', source=source, fill_alpha=.5)
    return plot


def static_histogram_and_label(col, label, n_bins=10,
                               col_max=None, col_min=None,
                               normalized=True):
    if col_max is None:
        col_max = col.max()
    if col_min is None:
        col_min = col.min()
    truncated = col[(col <= col_max) & (col >= col_min)]
    hist, edges = np.histogram(truncated, bins=n_bins, density=normalized)
    cols = pd.DataFrame({'col': col, 'label': label})

    label_hist = np.nan_to_num(cols['label'].groupby(
        pd.cut(col, edges, right=False)).sum().values, 0)
    if normalized:
        label_hist = label_hist / (label_hist.sum() * (edges[1] - edges[0]))
    source = ColumnDataSource(pd.DataFrame({'hist': hist,
                                            'left': edges[:-1],
                                            'right': edges[1:],
                                            'label': label_hist}))
    if normalized:
        hover = HoverTool(
            tooltips=[
                ("Bin", " [@left{0.00}, @right{0.00})"),
            ],
            mode='mouse')
    else:
        hover = HoverTool(
            tooltips=[
                ("Height", " @hist"),
                ("Bin", " [@left{0.00}, @right{0.00})"),
            ],
            mode='mouse')

    plot = figure(tools=[hover, 'box_zoom', 'save', 'reset'])
    plot.quad(top='hist', bottom=0, left='left',
              right='right', line_color='white',
              source=source, fill_alpha=.5)
    plot.quad(top='label', bottom=0, left='left',
              right='right', color='purple',
              line_color='white', source=source, fill_alpha=.5)
    return plot
