# -*- coding: utf-8 -*-

'''The plotting module.

Contents:
        feature_importances_plot
'''
import pandas as pd
import numpy as np

from bokeh.models import ColumnDataSource, HoverTool, Slider, RangeSlider
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.io import output_notebook

from math import pi

from bokeh.palettes import Category20

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


def _make_pie_source(col, mergepast=10, sort=True, drop_n=None):
    values = col.reset_index().groupby(col.name).count()
    total = float(col.shape[0])

    counts = values[values.columns[0]].tolist()
    percents = [x/total for x in counts]
    tmp = pd.DataFrame({'names': values.index,
                        'counts': counts,
                        'percents': percents})
    if sort:
        tmp = tmp.sort_values(by='counts', ascending=False)

    if drop_n:
        tmp = tmp.iloc[drop_n:]
        tmp['percents'] = tmp['percents']/tmp['percents'].sum()
    starts = []
    ends = []
    loc = 0
    for perc in tmp['percents']:
        starts.append(loc)
        loc += 2*pi*perc
        ends.append(loc)
    tmp['starts'] = starts
    tmp['ends'] = ends

    if mergepast is not None and mergepast < tmp.shape[0]:
        percent = tmp.iloc[mergepast:]['percents'].sum()
        count = tmp.iloc[mergepast:]['counts'].sum()
        start = tmp.iloc[mergepast:mergepast+1]['starts'].values
        end = tmp.iloc[-1:]['ends'].values
        tmp = pd.concat([tmp.iloc[:mergepast],
                         pd.DataFrame({'names': ['Other'],
                                       'counts': [count],
                                       'percents': [percent],
                                       'starts': start,
                                       'ends': end})])
    tmp['colors'] = [Category20[20][i % 20] for i, _ in enumerate(tmp['names'])]

    return tmp


def static_piechart(col, sort=True, mergepast=10, drop_n=None):
    source = ColumnDataSource(_make_pie_source(col, mergepast, sort, drop_n))

    plot = figure(height=500, toolbar_location=None)
    plot.wedge(x=0, y=0,
               radius=0.3,
               start_angle='starts',
               end_angle='ends',
               line_color='white',
               color='colors',
               legend='names',
               source=source)
    plot.axis.axis_label = None
    plot.axis.visible = False
    plot.grid.grid_line_color = None
    return plot


def dynamic_histogram(col):
    def modify_doc(doc, col):
        hover = HoverTool(
            tooltips=[
                ("Height", " @hist"),
                ("Bin", " [@left{0.00}, @right{0.00})"),
            ],
            mode='mouse')

        truncated = col[(col <= col.max()) & (col >= col.min())]
        hist, edges = np.histogram(truncated, bins=10, density=False)
        source = ColumnDataSource(pd.DataFrame({'hist': hist,
                                                'left': edges[:-1],
                                                'right': edges[1:]}
                                               ))

        plot = figure(tools=[hover, 'box_zoom', 'save', 'reset'])
        plot.quad(top='hist', bottom=0,
                  left='left', right='right',
                  line_color='white', source=source, fill_alpha=.5)

        def callback(attr, old, new):
            truncated = col[(col < range_select.value[1]) & (col > range_select.value[0])]
            hist, edges = np.histogram(truncated,
                                       bins=slider.value,
                                       density=False)

            source.data = ColumnDataSource(pd.DataFrame({'hist': hist,
                                                         'left': edges[:-1],
                                                         'right': edges[1:]})).data

        slider = Slider(start=1, end=100,
                        value=10, step=1,
                        title="Bins")
        slider.on_change('value', callback)

        range_select = RangeSlider(start=col.min(),
                                   end=col.max(),
                                   value=(col.min(), col.max()),
                                   step=5, title='Histogram Range')
        range_select.on_change('value', callback)

        doc.add_root(column(slider, range_select, plot))

    return lambda doc: modify_doc(doc, col)


def dynamic_histogram_and_label(col, label, normalized=True):
    def modify_doc(doc, col, label, normalized):

        truncated = col[(col <= col.max()) & (col >= col.min())]
        hist, edges = np.histogram(truncated, bins=10, density=normalized)
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

        def callback(attr, old, new):

            truncated = col[(col < range_select.value[1]) & (col > range_select.value[0])]

            hist, edges = np.histogram(truncated,
                                       bins=slider.value,
                                       density=normalized)
            label_hist = np.nan_to_num(cols['label'].groupby(
                pd.cut(col, edges, right=False)).sum().values, 0)

            if normalized:
                label_hist = label_hist / (label_hist.sum() * (edges[1] - edges[0]))

            source.data = ColumnDataSource(pd.DataFrame({'hist': hist,
                                                         'left': edges[:-1],
                                                         'right': edges[1:],
                                                         'label': label_hist})).data

        slider = Slider(start=1, end=100, value=10, step=1, title="Bins")
        slider.on_change('value', callback)

        range_select = RangeSlider(start=col.min(),
                                   end=col.max(),
                                   value=(col.min(), col.max()),
                                   step=5, title='Histogram Range')
        range_select.on_change('value', callback)

        doc.add_root(column(slider, range_select, plot))
    return lambda doc: modify_doc(doc, col, label, normalized)
