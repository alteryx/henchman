# -*- coding: utf-8 -*-

'''The plotting module.

Contents:
        feature_importances_plot
'''

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import output_notebook

from henchman.learning import _raw_feature_importances


def feature_importances_plot(X, model, n_feats=5):
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
