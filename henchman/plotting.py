# -*- coding: utf-8 -*-

'''The plotting module.

Contents:
        feature_importances_plot
'''
import pandas as pd
import numpy as np

from bokeh.models import (ColumnDataSource, HoverTool,
                          Slider, RangeSlider, CheckboxGroup,
                          Range1d, CDSView, Plot, MultiLine,
                          Circle, TapTool, BoxZoomTool, ResetTool, SaveTool)

from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges

from bokeh.layouts import column, row
from bokeh.plotting import figure

from bokeh.io import output_notebook
from bokeh.io.export import get_screenshot_as_png
import bokeh.io as io

from math import pi

from bokeh.palettes import Category20, Spectral4

from henchman.learning import _raw_feature_importances
from henchman.learning import create_model

from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score, roc_curve)

import networkx as nx


def show_template():
    '''Prints a template for `show`.
    '''
    print('show(plot,\n'
          '     png=False,\n'
          '     width=None,\n'
          '     height=None,\n'
          '     title=\'Temporary title\',\n'
          '     x_axis=\'my xaxis name\',\n'
          '     y_axis=\'my yaxis name\',\n'
          '     x_range=(0, 10) or None,\n'
          '     y_range=(0, 10) or None)\n')
    return None


def show(plot, png=False,
         width=None, height=None,
         title=None, x_axis=None, y_axis=None,
         x_range=None, y_range=None):
    '''Format and show a bokeh plot.
    This is a wrapper around bokeh show which can add common
    plot attributes like height, axis labels and whether or not
    you would like the output as a png. This function also runs
    the bokeh function ``output_notebook()`` to start.

    You can get a full list of options by function with ``show_template()``.

    Args:
        plot (bokeh.figure or doc): The plot to show.
        png (bool): If True, return a png of the plot. Default is False
        width (int, optional): Plot width.
        height (int, optional): Plot height.
        title (str, optional): The title for the plot.
        x_axis (str, optional): The x_axis label.
        y_axis (str, optional): The y_axis label.
        x_range (tuple[int, int], optional): A min and max x value to plot.
        y_range (tuple[int, int], optional): A min and max y value to plot.

    Example:
        >>> import henchman.plotting as hplot
        >>> hplot.show_template()
        show(plot,
             png=False,
             width=None,
             height=None,
             title='Temporary title',
             x_axis='my xaxis name',
             y_axis='my yaxis name',
             x_range=(0, 10) or None,
             y_range=(0, 10) or None)

        >>> hplot.show(plot, width=500, title='My Plot Title')
    '''
    output_notebook(hide_banner=True)
    if width is not None:
        plot.width = width
    if height is not None:
        plot.height = height

    if title is not None:
        plot.title.text = title
    if x_axis is not None:
        plot.xaxis.axis_label = x_axis
    if y_axis is not None:
        plot.yaxis.axis_label = y_axis

    if x_range is not None:
        plot.x_range = Range1d(x_range[0], x_range[1])
    if y_range is not None:
        plot.y_range = Range1d(y_range[0], y_range[1])

    if png:
        return get_screenshot_as_png(plot, driver=None)

    return io.show(plot)


def feature_importances(X, model, n_feats=5):
    '''Plot feature importances.

    Args:
        X (pd.DataFrame): A dataframe with which you have trained.
        model: Any fit model with a ``feature_importances_`` attribute.
        n_feats (int): The number of features to plot.

    Example:
        >>> import henchman.plotting as hplot
        >>> plot = hplot.feature_importances(X, model, n_feats=10)
        >>> hplot.show(plot)

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
    '''Creates a static bokeh histogram.
    User can modify the number of bins and column bounds.

    Args:
        col (pd.Series): The column from which to make a histogram.
        n_bins (int): The number of bins of the histogram.
        col_max (float): Maximum value to include in histogram.
        col_min (float): Minimum value to include in histogram

    Example:
        If the dataframe ``X`` has a column named ``amount``:

        >>> import henchman.plotting as hplot
        >>> plot = hplot.static_histogram(X['amount'])
        >>> hplot.show(plot)
    '''
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
    '''Creates a static bokeh histogram with binary label.
    You can use this function to see how a binary label compares with
    a particular attribute. Can set number of bins, column bounds
    and whether or not to normalize both functions. Normalizing will
    lose exact values but can sometimes make the columns easier
    to compare.

    Args:
        col (pd.Series): The column from which to make a histogram.
        label (pd.Series): A binary label that you would like to track.
        n_bins (int): The number of bins of the histogram.
        col_max (float): Maximum value to include in histogram.
        col_min (float): Minimum value to include in histogram

    Example:
        If the dataframe ``X`` has a column named ``amount`` and
        a label ``y``, you can compare them with

        >>> import henchman.plotting as hplot
        >>> plot1 = hplot.static_histogram_and_label(X['amount'], y)
        >>> hplot.show(plot1)

        If you want the raw number of positive labels in each bin, set normalized

        >>> plot2 = hplot.static_histogram_and_label(X['amount'], y, normalized=False)
        >>> hplot.show(plot2)
    '''
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
    tmp['colors'] = [Category20[20][i % 20]
                     for i, _ in enumerate(tmp['names'])]

    return tmp


def static_piechart(col, sort=True, mergepast=10, drop_n=None, hover=True):
    '''Creates a static piechart.
    Finds all of the unique values in a column and makes a piechart
    out of them. By default, the chart will be sorted and merge together
    any values past the 10th most common.

    Args:
        col (pd.Series): The column from which to make the piechart.
        sort (bool): Whether or not to sort by frequency. Default is True.
        mergepast (int): Merge infrequent column values. Default is 10.
        drop_n (int): How many high frequency values to drop. Default is None.

    Example:
        If the dataframe ``X`` has a column named ``car_color``:

        >>> import henchman.plotting as hplot
        >>> plot = hplot.static_piechart(X['car_color'], sort=False, mergepast=None)
        >>> hplot.show(plot)
    '''

    source = ColumnDataSource(_make_pie_source(col, mergepast, sort, drop_n))
    tools = ['box_zoom', 'save', 'reset']
    if hover:
        hover = HoverTool(
            tooltips=[
                ("Name", " @names"),
            ],
            mode='mouse')
        tools = tools + [hover]
    plot = figure(height=500, tools=tools, toolbar_location='above')
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


def _make_scatter_source(col1, col2):
    tmp = pd.DataFrame({col1.name: col1, col2.name: col2})
    tmp['pairs'] = tmp.apply(lambda row: (row[0], row[1]), axis=1)
    source = pd.DataFrame(tmp.groupby('pairs').first())
    source['count'] = tmp.groupby('pairs').count().iloc[:, 1]
    source['x'] = source[col1.name]
    source['y'] = source[col2.name]
    return source


def static_scatterplot(col1, col2, hover=True):
    '''Creates a static scatterplot.
    Plots two numeric variables against one another. In this function,
    we only take one from each numeric pair and count how many times it
    appears in the data.


    Args:
        col1 (pd.Series): The column to use for the x_axis.
        col2 (pd.Series): The column to use for the y_axis.
        hover (bool): Whether or not to include the hover tooltip. Default is True.

    Example:
        If the dataframe ``X`` has columns named ``amount`` and ``num_purchases``:

        >>> import henchman.plotting as hplot
        >>> plot = hplot.static_scatterplot(X['num_purchases'], X['amount'])
        >>> hplot.show(plot)
    '''
    source = ColumnDataSource(_make_scatter_source(col1, col2))
    tools = ['box_zoom', 'save', 'reset']
    if hover:
        hover = HoverTool(tooltips=[
            (col1.name, '@x'),
            (col2.name, '@y'),
            ('count', '@count'),
        ])
        tools += [hover]

    p = figure(tools=tools)
    p.scatter(x='x',
              y='y',
              source=source,
              alpha=.5)
    return p


def _make_scatter_label_source(col1, col2, label):
    tmp = pd.DataFrame({col1.name: col1, col2.name: col2, label.name: label})
    tmp['pairs'] = tmp.apply(lambda row: (row[0], row[1], row[2]), axis=1)
    source = tmp

    source['x'] = source[col1.name]
    source['y'] = source[col2.name]
    source['label'] = source[label.name]

    label_to_int = {name: i for i, name in enumerate(label.unique())}
    colors = [Category20[20][label_to_int[value] % 20 + 1] for value in source['label']]

    source['color'] = colors
    return source


def static_scatterplot_and_label(col1, col2, label, hover=False):
    '''Creates a static scatterplot with label information.
    Plots two numeric variables against one another colors
    the results by a binary label. This can give information on if
    these two columns are related to a label. Unlike static_scatterplot,
    this function does not start by reducing to unique values.
    Use the hovertool at your own risk.

    Args:
        col1 (pd.Series): The column to use for the x_axis.
        col2 (pd.Series): The column to use for the y_axis.
        label (pd.Series): The binary label.
        hover (bool): Whether or not to include the hover tooltip. Default is False.

    Example:
        If the dataframe ``X`` has columns named ``amount``
        and ``num_purchases`` with a binary label ``y``:

        >>> import henchman.plotting as hplot
        >>> plot = hplot.static_scatterplot_and_label(X['num_purchases'], X['amount'], y)
        >>> hplot.show(plot)
    '''
    source = ColumnDataSource(_make_scatter_label_source(col1, col2, label))
    tools = ['box_zoom', 'save', 'reset']
    if hover:
        hover = HoverTool(tooltips=[
            (col1.name, '@x'),
            (col2.name, '@y'),
            (label.name, '@label'),
        ])
        tools += [hover]

    plot = figure(tools=tools)
    plot.scatter(x='x',
                 y='y',
                 color='color',
                 legend='label',
                 source=source,
                 alpha=.8)
    return plot


def dynamic_histogram(col):
    '''Creates a dynamic histogram.
    Allows for interactive modification of the static_histogram plot.

    Args:
        col (pd.Series): The column from which to make the histogram.

    Example:
        If the dataframe ``X`` has a column named ``amount``.

        >>> import henchman.plotting as hplot
        >>> plot = hplot.dynamic_histogram(X['amount'])
        >>> hplot.show(plot1)
    '''
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
            truncated = col[(col < range_select.value[1]) &
                            (col > range_select.value[0])]
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
    '''Creates a dynamic histogram with binary label.
    This function builds the static_histogram_and_label, but allows
    for modification of the parameters.

    Args:
        col (pd.Series): The column from which to make a histogram.
        label (pd.Series): The binary label you'd like to track
        normalized (bool): Whether or not to normalize both histograms.
                Default values is ``True``.

    Examples:
        If the dataframe ``X`` has a column named ``amount`` and
        a label ``y``, you can compare them with

        >>> import henchman.plotting as hplot
        >>> plot1 = hplot.dynamic_histogram_and_label(X['amount'], y)
        >>> hplot.show(plot1)

        If you want the raw number of positive labels in each bin, set normalized

        >>> plot2 = hplot.dynamic_histogram_and_label(X['amount'], y, normalized=False)
        >>> hplot.show(plot2)

    '''
    def modify_doc(doc, col, label, normalized):

        truncated = col[(col <= col.max()) & (col >= col.min())]
        hist, edges = np.histogram(truncated, bins=10, density=normalized)
        cols = pd.DataFrame({'col': col, 'label': label})

        label_hist = np.nan_to_num(cols['label'].groupby(
            pd.cut(col, edges, right=False)).sum().values, 0)
        if normalized:
            label_hist = label_hist / \
                (label_hist.sum() * (edges[1] - edges[0]))
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

            truncated = col[(col < range_select.value[1]) &
                            (col > range_select.value[0])]

            hist, edges = np.histogram(truncated,
                                       bins=slider.value,
                                       density=normalized)
            label_hist = np.nan_to_num(cols['label'].groupby(
                pd.cut(col, edges, right=False)).sum().values, 0)

            if normalized:
                label_hist = label_hist / \
                    (label_hist.sum() * (edges[1] - edges[0]))

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


def dynamic_piechart(col, hover=True):
    '''Creates a dynamic piechart.
    This allows the user to interactively change the arguments
    to a static piechart.

    Args:
        col (pd.Series): The column from which to make the piechart.

    Example:
        If the dataframe ``X`` has a column named ``car_color``:

        >>> import henchman.plotting as hplot
        >>> plot = hplot.dynamic_piechart(X['car_color'])
        >>> hplot.show(plot)
    '''
    def modify_doc(doc, col, hover):
        n_values = col.nunique()
        source = ColumnDataSource(_make_pie_source(col,
                                                   mergepast=n_values))
        tools = ['box_zoom', 'save', 'reset']
        if hover:
            hover = HoverTool(
                tooltips=[
                    ("Name", " @names"),
                ],
                mode='mouse')
            tools = tools + [hover]
        plot = figure(height=500, tools=tools, toolbar_location='above')
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

        def callback(attr, old, new):

            source.data = ColumnDataSource(
                _make_pie_source(col,
                                 sort=sorted_button.active,
                                 mergepast=merge_slider.value,
                                 drop_n=drop_slider.value)).data
        sorted_button = CheckboxGroup(
            labels=["Sorted"], active=[0, 1])
        sorted_button.on_change('active', callback)

        merge_slider = Slider(start=1, end=n_values,
                              value=n_values, step=1,
                              title="Merge Slider")
        merge_slider.on_change('value', callback)
        drop_slider = Slider(start=0, end=n_values,
                             value=0, step=1,
                             title="Drop Slider")
        drop_slider.on_change('value', callback)

        doc.add_root(
            column(
                row(
                    column(merge_slider, drop_slider), sorted_button
                ),
                plot
            ))

    return lambda doc: modify_doc(doc, col, hover)


def roc_auc(X, y, model, pos_label=1, prob_col=1, n_splits=1):
    '''Plots the reveiver operating characteristic curve.
    This function creates a fit model and shows the results of the roc curve.

    Args:
        X (pd.DataFrame): The dataframe on which to create a model.
        y (pd.Series): The labels for which to create a model.
        pos_label (int): Which label to check for fpr and tpr. Default is 1.
        prob_col (int): The columns of the probs dataframe to use.
        n_splits (int): The number of splits to use in validation.

    Example:
        If the dataframe ``X`` has a binary classification label y:

        >>> import henchman.plotting as hplot
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> plot = hplot.roc_auc(X, y, RandomForestClassifier())
        >>> hplot.show(plot)
    '''

    scores, model, df_list = create_model(X, y, model, roc_auc_score, _return_df=True, n_splits=n_splits)

    probs = model.predict_proba(df_list[1])
    fpr, tpr, thresholds = roc_curve(df_list[3],
                                     probs[:, prob_col],
                                     pos_label=pos_label)
    tools = ['box_zoom', 'save', 'reset']
    plot = figure(tools=tools)
    plot.line(x=fpr, y=tpr)
    plot.title.text = 'Receiver operating characteristic'
    plot.xaxis.axis_label = 'False Positive Rate'
    plot.yaxis.axis_label = 'True Positive Rate'

    plot.line(x=fpr, y=fpr, color='red', line_dash='dashed')
    return(plot)


def f1(X, y, model, n_precs=1000, n_splits=1):
    '''Plots the precision, recall and f1 at various thresholds.
    This function creates a fit model and shows the precision,
    recall and f1 results at multiple thresholds.

    Args:
        X (pd.DataFrame): The dataframe on which to create a model.
        y (pd.Series): The labels for which to create a model.
        n_precs (int): The number of thresholds to sample between 0 and 1.
        n_splits (int): The number of splits to use in validation.

    Example:
        If the dataframe ``X`` has a binary classification label ``y``:

        >>> import henchman.plotting as hplot
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> plot = hplot.f1(X, y, RandomForestClassifier())
        >>> hplot.show(plot)
    '''

    scores, model, df_list = create_model(X, y, model, roc_auc_score, _return_df=True, n_splits=n_splits)
    probs = model.predict_proba(df_list[1])
    threshes = [x/float(n_precs) for x in range(0, n_precs)]
    precisions = [precision_score(df_list[3], probs[:, 1] > t) for t in threshes]
    recalls = [recall_score(df_list[3], probs[:, 1] > t) for t in threshes]
    fones = [f1_score(df_list[3], probs[:, 1] > t) for t in threshes]

    tools = ['box_zoom', 'save', 'reset']

    plot = figure(tools=tools)
    plot.line(x=threshes, y=precisions, color='green', legend='precision')
    plot.line(x=threshes, y=recalls, color='blue', legend='recall')
    plot.line(x=threshes, y=fones, color='red', legend='f1')

    plot.xaxis.axis_label = 'Threshold'
    plot.title.text = 'Precision, Recall, and F1 by Threshold'
    return(plot)


def dendrogram(D):
    '''Creates a dynamic dendrogram plot.
    This plot can show full structure of a given dendrogram.

    Args:
        D (henchman.selection.Dendrogram): An initialized dendrogram object

    Examples:
        >>> from henchman.selection import Dendrogram
        >>> from henchman.plotting import show
        >>> import henchman.plotting as hplot
        >>> D = Dendrogram(X)
        >>> plot = hplot.dynamic_dendrogram(D)
        >>> show(plot)
    '''
    def modify_doc(doc, D):
        G = nx.Graph()

        vertices_source = ColumnDataSource(
            pd.DataFrame({'index': D.columns.keys(),
                          'desc': D.columns.values()}))
        edges_source = ColumnDataSource(
            pd.DataFrame(D.edges[0]).rename(
                columns={1: 'end', 0: 'start'}))
        step_source = ColumnDataSource(
            pd.DataFrame({'step': [0],
                          'thresh': [D.threshlist[0]],
                          'components': [len(D.graphs[0])]}))

        G.add_nodes_from(vertices_source.data['index'])
        G.add_edges_from(zip(
            edges_source.data['start'],
            edges_source.data['end']))

        graph_renderer = from_networkx(G, nx.circular_layout,
                                       scale=1, center=(0, 0))

        graph_renderer.node_renderer.data_source = vertices_source
        graph_renderer.node_renderer.view = CDSView(source=vertices_source)
        graph_renderer.edge_renderer.data_source = edges_source
        graph_renderer.edge_renderer.view = CDSView(source=edges_source)

        plot = Plot(plot_width=400, plot_height=400,
                    x_range=Range1d(-1.1, 1.1),
                    y_range=Range1d(-1.1, 1.1))
        plot.title.text = "Feature Connectivity"

        graph_renderer.node_renderer.glyph = Circle(
            size=5, fill_color=Spectral4[0])
        graph_renderer.node_renderer.selection_glyph = Circle(
            size=15, fill_color=Spectral4[2])

        graph_renderer.edge_renderer.data_source = edges_source
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC",
                                                       line_alpha=0.6,
                                                       line_width=.5)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(
            line_color=Spectral4[2],
            line_width=3)

        graph_renderer.node_renderer.hover_glyph = Circle(
            size=5,
            fill_color=Spectral4[1])

        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = NodesAndLinkedEdges()

        plot.renderers.append(graph_renderer)

        plot.add_tools(
            HoverTool(tooltips=[("feature", "@desc"),
                                ("index", "@index"), ]),
            TapTool(),
            BoxZoomTool(),
            SaveTool(),
            ResetTool())

        data_table = DataTable(source=step_source,
                               columns=[TableColumn(field='step',
                                                    title='Step'),
                                        TableColumn(field='thresh',
                                                    title='Thresh'),
                                        TableColumn(field='components',
                                                    title='Components')],
                               height=50, width=400)

        def callback(attr, old, new):
            edges = D.edges[slider.value]
            edges_source.data = ColumnDataSource(
                pd.DataFrame(edges).rename(columns={1: 'end',
                                                    0: 'start'})).data
            step_source.data = ColumnDataSource(
                {'step': [slider.value],
                 'thresh': [D.threshlist[slider.value]],
                 'components': [len(D.graphs[slider.value])]}).data

        slider = Slider(start=0,
                        end=len(D.edges),
                        value=0,
                        step=1,
                        title="Step")
        slider.on_change('value', callback)

        doc.add_root(column(slider, data_table, plot))
    return lambda doc: modify_doc(doc, D)
