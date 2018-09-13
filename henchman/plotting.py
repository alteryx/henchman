# -*- coding: utf-8 -*-

'''The plotting module.

Contents:
        show_template
        show
        piechart
        histogram
        scatter
        timeseries
        dendrogram
        feature_importances
'''
import pandas as pd
import numpy as np

from bokeh.models import (ColumnDataSource, HoverTool,
                          Slider, RangeSlider, CheckboxGroup, DateRangeSlider,
                          Range1d, CDSView, Plot, MultiLine,
                          Circle, TapTool, BoxZoomTool, ResetTool, SaveTool)

from bokeh.models.widgets import DataTable, TableColumn, Dropdown
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges

from bokeh.layouts import column, row
import bokeh.layouts as layouts

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
    '''Prints a template for `show`. See :func:`show <henchman.plotting.show>` for details.

    Example:
        >>> import henchman.plotting as hplot
        >>> hplot.show_template()
    '''
    print('show(plot,\n'
          '     static=False,\n'
          '     png=False,\n'
          '     hover=False,\n'
          '     colors=None,\n'
          '     width=None,\n'
          '     height=None,\n'
          '     title=\'Temporary title\',\n'
          '     x_axis=\'my xaxis name\',\n'
          '     y_axis=\'my yaxis name\',\n'
          '     x_range=(0, 10) or None,\n'
          '     y_range=(0, 10) or None)\n')
    return None


def _modify_plot(plot, figargs):
    '''Add text and modify figure attributes. This is an internal
    function which allows for figure attributes to be passed into
    interactive functions.

    Args:
        plot (bokeh.figure): The figure to modify
        figargs (dict[assorted]): A dictionary of width, height,
                title, x_axis, y_axis, x_range and y_range.

    '''
    if figargs['width'] is not None:
        plot.width = figargs['width']
    if figargs['height'] is not None:
        plot.height = figargs['height']

    if figargs['title'] is not None:
        plot.title.text = figargs['title']
    if figargs['x_axis'] is not None:
        plot.xaxis.axis_label = figargs['x_axis']
    if figargs['y_axis'] is not None:
        plot.yaxis.axis_label = figargs['y_axis']

    if figargs['x_range'] is not None:
        plot.x_range = Range1d(figargs['x_range'][0], figargs['x_range'][1])
    if figargs['y_range'] is not None:
        plot.y_range = Range1d(figargs['y_range'][0], figargs['y_range'][1])

    return plot


def show(plot, png=False, static=False, hover=True,
         width=None, height=None,
         title=None, x_axis=None, y_axis=None,
         x_range=None, y_range=None, colors=None,
         fig=False):
    '''Format and show a bokeh plot.
    This is a wrapper around bokeh show which can add common
    plot attributes like height, axis labels and whether or not
    you would like the output as a png. This function also runs
    the bokeh function ``output_notebook()`` to start.

    You can get a full list of options by function with ``show_template()``.

    Args:
        plot (function): The plot to show.
        static (bool): If True, show a static bokeh plot.
        hover (bool): If True, show the hovertool. Default is True.
        width (int, optional): Plot width.
        height (int, optional): Plot height.
        title (str, optional): The title for the plot.
        x_axis (str, optional): The x_axis label.
        y_axis (str, optional): The y_axis label.
        x_range (tuple[int, int], optional): A min and max x value to plot.
        y_range (tuple[int, int], optional): A min and max y value to plot.
        colors (list[str], optional): A list of colors to use for the plot.
        png (bool): If True, return a png of the plot. Default is False
        fig (bool, advanced): If True, return a bokeh figure instead of
            showing the plot. Only use if you want to manipulate the bokeh figure directly.

    Example:
        >>> import henchman.plotting as hplot
        >>> hplot.show_template()
        show(plot,
             static=False,
             png=False,
             hover=True,
             width=None,
             height=None,
             title='Temporary title',
             x_axis='my xaxis name',
             y_axis='my yaxis name',
             x_range=(0, 10) or None,
             y_range=(0, 10) or None,
             colors=None)

        >>> hplot.show(plot, width=500, title='My Plot Title')
        >>> hplot.show(plot, png=True, static=True)
    '''
    output_notebook(hide_banner=True)
    figargs = {'static': static, 'png': png, 'hover': hover,
               'width': width, 'height': height,
               'title': title, 'x_axis': x_axis, 'y_axis': y_axis,
               'x_range': x_range, 'y_range': y_range,
               'colors': colors}
    figure = plot(figargs=figargs)

    if fig:
        return figure

    if png:
        figargs['static'] = True
        return get_screenshot_as_png(plot(figargs=figargs), driver=None)

    return io.show(figure)


def gridplot(plots, n_cols=1):
    '''Create a gridplot.
    This is a wrapper around bokeh gridplot meant to easily work with
    henchman plots. Note that the figures must be ``static`` for this to work.
    This function call is a work in progress and will likely be depreciated in
    favor of something stable.

    Args:
        plots (list[bokeh.figure]): The plots to show. Either a list or a list of lists.
        n_cols (int): The number of columns. This will be ignored if a list of lists is passed in.

    Example:
        >>> import henchman.plotting as hplot

        >>> p1 = hplot.show(plot, static=True, fig=True)
        >>> p2 = hplot.show(plot, static=True, fig=True)
        >>> hplot.gridplot([p1, p2], n_cols=2)
    '''
    output_notebook(hide_banner=True)
    if isinstance(plots[0], list):
        return io.show(layouts.gridplot(plots))
    return io.show(layouts.gridplot(plots, ncols=n_cols))


def piechart(col, sort=True, mergepast=None, drop_n=None, figargs=None):
    '''Creates a piechart.
    Finds all of the unique values in a column and makes a piechart
    out of them. By default, this will make a dynamic piechart with
    sliders for the different parameters.

    Args:
        col (pd.Series): The column from which to make the piechart.
        sort (bool): Whether or not to sort by frequency for static plot. Default is True.
        mergepast (int): Merge infrequent column values for static plot. Default is 10.
        drop_n (int): How many high frequency values to drop for static plot. Default is None.

    Example:
        If the dataframe ``X`` has a column named ``car_color``:

        >>> import henchman.plotting as hplot
        >>> plot = hplot.piechart(X['car_color'])
        >>> hplot.show(plot)

        For a static plot:

        >>> import henchman.plotting as hplot
        >>> plot = hplot.piechart(X['car_color'], sort=False)
        >>> hplot.show(plot, static=True)
    '''
    if figargs is None:
        return lambda figargs: piechart(col, sort, mergepast, drop_n, figargs)

    source = ColumnDataSource(_make_piechart_source(col, mergepast, sort, drop_n, figargs))
    plot = _make_piechart_plot(source, figargs)
    plot = _modify_plot(plot, figargs)

    if figargs['static']:
        return plot

    def modify_doc(doc, col, sort, mergepast, drop_n, figargs):
        def callback(attr, old, new):
            try:
                source.data = ColumnDataSource(
                    _make_piechart_source(col,
                                          sort=sorted_button.active,
                                          mergepast=merge_slider.value,
                                          drop_n=drop_slider.value,
                                          figargs=figargs)).data
            except Exception as e:
                print(e)

        sorted_button, merge_slider, drop_slider = _piechart_widgets(
            col, sort, mergepast, drop_n, callback)

        doc.add_root(
            column(row(column(merge_slider, drop_slider), sorted_button), plot))

    return lambda doc: modify_doc(doc, col, sort, mergepast, drop_n, figargs)


def histogram(col, y=None, n_bins=10, col_max=None, col_min=None,
              normalized=False, figargs=None):
    '''Creates a histogram.
    This function takes a single input and creates a histogram from it.
    There is an optional second column input for labels, if you would
    like to see how a label is distributed relative to your numeric
    variable.

    Args:
        col (pd.Series): The column from which to make a histogram.
        y (pd.Series, optional): A binary label that you would like to track.
        n_bins (int): The number of bins of the histogram. Default is 10.
        col_max (float): Maximum value to include in histogram.
        col_min (float): Minimum value to include in histogram.
        normalized (bool): Whether or not to normalize the columns. Default is False.

    Example:
        If the dataframe ``X`` has a column named ``amount`` and
        a label ``y``, you can compare them with

        >>> import henchman.plotting as hplot
        >>> plot1 = hplot.histogram(X['amount'], y, normalized=False)
        >>> hplot.show(plot1)

        If you wanted a single variable histogram instead, omit y:

        >>> plot2 = hplot.histogram(X['amount'], col_max=200, n_bins=20)
        >>> hplot.show(plot2)
    '''
    if figargs is None:
        return lambda figargs: histogram(
            col, y, n_bins, col_max, col_min,
            normalized, figargs=figargs)

    source = ColumnDataSource(_make_histogram_source(col, y, n_bins, col_max, col_min, normalized))
    plot = _make_histogram_plot(y, source, figargs)
    plot = _modify_plot(plot, figargs)

    if figargs['static']:
        return plot

    def modify_doc(doc, col, y, n_bins, col_max, col_min, normalized, figargs):
        def callback(attr, old, new):
            try:
                source.data = ColumnDataSource(_make_histogram_source(
                    col, y, n_bins=slider.value, col_max=range_select.value[1],
                    col_min=range_select.value[0], normalized=normalized)).data
            except Exception as e:
                print(e)

        slider, range_select = _histogram_widgets(col, y, n_bins, col_max, col_min, callback)

        doc.add_root(column(slider, range_select, plot))
    return lambda doc: modify_doc(doc, col, y, n_bins, col_max, col_min, normalized, figargs)


def timeseries(col_1, col_2, col_max=None, col_min=None, n_bins=10,
               aggregate='mean', figargs=None):
    '''Creates a time based aggregations of a numeric variable.
    This function allows for the user to mean, count, sum or find the min
    or max of a second variable with regards to a timeseries.

    Args:
        col_1 (pd.Series): The column from which to create bins. Must be a datetime.
        col_2 (pd.Series): The column to aggregate.
        col_max (pd.datetime): The maximum value for the x-axis. Default is None.
        col_min (pd.datetime): The minimum value for the x-axis. Default is None.
        n_bins (int): The number of time bins to make.
        aggregate (str): What aggregation to do on the numeric column. Options are
            'mean', 'sum', 'count', 'max' and 'min'. Default is 'mean'.

    Example:
        If the dataframe ``X`` has a columns named ``amount`` and ``date``.

        >>> import henchman.plotting as hplot
        >>> plot = hplot.timeseries(X['date'], X['amount'])
        >>> hplot.show(plot)

        For a bokeh plot without sliders:

        >>> plot2 = hplot.timeseries(X['date'], X['amount'], n_bins=50)
        >>> hplot.show(plot2, static=True)
    '''
    if figargs is None:
        return lambda figargs: timeseries(col_1, col_2, col_max, col_min,
                                          n_bins, aggregate, figargs=figargs)

    source = ColumnDataSource(_make_timeseries_source(col_1, col_2, col_max,
                                                      col_min, n_bins, aggregate))
    plot = _make_timeseries_plot(source, figargs)
    plot = _modify_plot(plot, figargs)

    if figargs['static']:
        return plot

    def modify_doc(doc, col_1, col_2, col_max, col_min, n_bins, aggregate, figargs):
        def callback(attr, old, new):
            try:
                source.data = ColumnDataSource(
                    _make_timeseries_source(col_1, col_2,
                                            col_max=range_select.value_as_datetime[1],
                                            col_min=range_select.value_as_datetime[0],
                                            n_bins=slider.value,
                                            aggregate=dropdown.value)).data
                dropdown.label = dropdown.value
            except Exception as e:
                print(e)

        slider, range_select, dropdown = _timeseries_widgets(
            col_1, col_2, col_max, col_min, n_bins, aggregate, callback)
        doc.add_root(column(slider, range_select, dropdown, plot))

    return lambda doc: modify_doc(
        doc, col_1, col_2, col_max, col_min, n_bins, aggregate, figargs)


def scatter(col_1, col_2, cat=None, label=None, aggregate='last',
            figargs=None):
    '''Creates a scatter plot of two variables.
    This function allows for the display of two variables with
    an optional argument to groupby. By default, this
    allows for the user to see what two variable looks like as
    grouped by another. A standard example would be to look at
    the "last" row for a column that's changing over time.

    Args:
        col_1 (pd.Series): The x-values of the plotted points.
        col_2 (pd.Series): The y-values of the plotted points.
        cat (pd.Series, optional): A categorical variable to aggregate by.
        label (pd.Series, optional): A numeric label to be used in the hovertool.
        aggregate (str): The aggregation to use. Options are 'mean', 'last', 'sum', 'max' and 'min'.

    Example:
        If the dataframe ``X`` has a columns named ``amount`` and ``quantity``.

        >>> import henchman.plotting as hplot
        >>> plot = hplot.scatter(X['amount'], X['quantity'])
        >>> hplot.show(plot)

        If you would like to see the amount, quantity pair as aggregated by the ``month`` column:

        >>> plot2 = hplot.scatter(X['date'], X['amount'], cat=X['month'], aggregate='mean')
        >>> hplot.show(plot2)
    '''
    if figargs is None:
        return lambda figargs: scatter(
            col_1, col_2, cat, label, aggregate, figargs=figargs)
    source = ColumnDataSource(_make_scatter_source(col_1, col_2, cat, label, aggregate))
    plot = _make_scatter_plot(col_1, col_2, label, cat, source, figargs)
    plot = _modify_plot(plot, figargs)

    if figargs['static']:
        return plot

    def modify_doc(doc, col_1, col_2, cat, label, aggregate, figargs):
        def callback(attr, old, new):
            try:
                source.data = ColumnDataSource(
                    _make_scatter_source(col_1, col_2, cat, label, aggregate=dropdown.value)).data
                dropdown.label = dropdown.value
            except Exception as e:
                print(e)

        dropdown = _scatter_widgets(col_1, col_2, aggregate, callback)
        if cat is not None:
            doc.add_root(column(dropdown, plot))
        else:
            doc.add_root(plot)
    return lambda doc: modify_doc(doc, col_1, col_2, cat, label, aggregate, figargs)


def feature_importances(X, model, n_feats=5, figargs=None):
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
    if figargs is None:
        return lambda figargs: feature_importances(X, model, n_feats, figargs=figargs)
    feature_imps = _raw_feature_importances(X, model)
    features = [f[1] for f in feature_imps[0:n_feats]][::-1]
    importances = [f[0] for f in feature_imps[0:n_feats]][::-1]

    output_notebook()
    source = ColumnDataSource(data={'feature': features,
                                    'importance': importances})
    plot = figure(y_range=features,
                  height=500,
                  title="Random Forest Feature Importances")
    plot.hbar(y='feature',
              right='importance',
              height=.8,
              left=0,
              source=source,
              color="#008891")
    plot.toolbar_location = None
    plot.yaxis.major_label_text_font_size = '10pt'

    plot = _modify_plot(plot, figargs)
    return plot


def roc_auc(X, y, model, pos_label=1, prob_col=1, n_splits=1, figargs=None):
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
    if figargs is None:
        return lambda figargs: roc_auc(X, y, model, pos_label,
                                       prob_col, n_splits, figargs=figargs)
    (scores, model), df_list = create_model(
        X, y, model, roc_auc_score, _return_df=True, n_splits=n_splits)

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
    plot = _modify_plot(plot, figargs)
    return(plot)


def dendrogram(D, figargs=None):
    '''Creates a dendrogram plot.
    This plot can show full structure of a given dendrogram.

    Args:
        D (henchman.selection.Dendrogram): An initialized dendrogram object

    Examples:
        >>> from henchman.selection import Dendrogram
        >>> from henchman.plotting import show
        >>> import henchman.plotting as hplot
        >>> D = Dendrogram(X)
        >>> plot = hplot.dendrogram(D)
        >>> show(plot)
    '''
    if figargs is None:
        return lambda figargs: dendrogram(D, figargs=figargs)
    G = nx.Graph()

    vertices_source = ColumnDataSource(
        pd.DataFrame({'index': D.columns.keys(),
                      'desc': list(D.columns.values())}))
    edges_source = ColumnDataSource(
        pd.DataFrame(D.edges[0]).rename(
            columns={1: 'end', 0: 'start'}))
    step_source = ColumnDataSource(
        pd.DataFrame({'step': [0],
                      'thresh': [D.threshlist[0]],
                      'components': [len(D.graphs[0])]}))

    G.add_nodes_from([str(x) for x in vertices_source.data['index']])
    G.add_edges_from(zip(
        [str(x) for x in edges_source.data['start']],
        [str(x) for x in edges_source.data['end']]))

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

    plot = _modify_plot(plot, figargs)

    if figargs['static']:
        return plot

    def modify_doc(doc, D, figargs):
        data_table = DataTable(source=step_source,
                               columns=[TableColumn(field='step',
                                                    title='Step'),
                                        TableColumn(field='thresh',
                                                    title='Thresh'),
                                        TableColumn(field='components',
                                                    title='Components')],
                               height=50, width=400)

        def callback(attr, old, new):
            try:
                edges = D.edges[slider.value]
                edges_source.data = ColumnDataSource(
                    pd.DataFrame(edges).rename(columns={1: 'end',
                                                        0: 'start'})).data
                step_source.data = ColumnDataSource(
                    {'step': [slider.value],
                     'thresh': [D.threshlist[slider.value]],
                     'components': [len(D.graphs[slider.value])]}).data
            except Exception as e:
                print(e)

        slider = Slider(start=0,
                        end=(len(D.edges) - 1),
                        value=0,
                        step=1,
                        title="Step")
        slider.on_change('value', callback)

        doc.add_root(column(slider, data_table, plot))
    return lambda doc: modify_doc(doc, D, figargs)


def f1(X, y, model, n_precs=1000, n_splits=1, figargs=None):
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
    if figargs is None:
        return lambda figargs: f1(X, y, model, n_precs,
                                  n_splits, figargs=figargs)

    (scores, model), df_list = create_model(
        X, y, model, roc_auc_score, _return_df=True, n_splits=n_splits)
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
    plot = _modify_plot(plot, figargs)

    return(plot)


# Piechart Utilities #


def _make_piechart_source(col, mergepast=None, sort=True, drop_n=None, figargs=None):
    if mergepast is None:
        mergepast = col.nunique()
    values = col.reset_index().groupby(col.name).count()
    total = float(col.shape[0])

    counts = values[values.columns[0]].tolist()
    percents = [x / total for x in counts]
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

    if mergepast < tmp.shape[0]:
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
    if figargs['colors'] is None:
        figargs['colors'] = Category20[20]
    tmp['colors'] = [figargs['colors'][i % (len(figargs['colors'])-1)]
                     for i, _ in enumerate(tmp['names'])]

    return tmp


def _make_piechart_plot(source, figargs):
    tools = ['box_zoom', 'save', 'reset']
    if figargs['hover']:
        hover = HoverTool(
            tooltips=[
                ("Name", " @names"),
                ("Count", " @counts"),
                ("Percent", " @percents{0%}"),
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


def _piechart_widgets(col, sort, mergepast, drop_n, callback):
    if sort:
        active = [0]
    else:
        active = []
    sorted_button = CheckboxGroup(
        labels=["Sorted"], active=active)
    sorted_button.on_change('active', callback)

    merge_slider = Slider(start=1, end=col.nunique(),
                          value=mergepast or col.nunique(), step=1,
                          title="Merge Slider")
    merge_slider.on_change('value', callback)
    drop_slider = Slider(start=0, end=col.nunique(),
                         value=drop_n or 0, step=1,
                         title="Drop Slider")
    drop_slider.on_change('value', callback)
    return sorted_button, merge_slider, drop_slider


# Timeseries Utilities #


def _make_timeseries_source(col_1, col_2, col_max=None, col_min=None, n_bins=10, aggregate='mean'):
    col_1_time = pd.to_datetime(col_1)
    if col_max is None:
        col_max = col_1_time.max()
    if col_min is None:
        col_min = col_1_time.min()

    truncated = col_1_time[(col_1_time <= col_max) & (col_1_time >= col_min)]
    tmp = pd.DataFrame({col_1.name: truncated,
                        'height': col_2,
                        'splits': pd.cut(pd.to_numeric(truncated), n_bins, right=False)})

    tmp = tmp.groupby('splits')['height'].aggregate(aggregate).reset_index()
    tmp['left'] = list(tmp['splits'].apply(lambda x: pd.to_datetime(x.left)))
    tmp['right'] = list(tmp['splits'].apply(lambda x: pd.to_datetime(x.right)))
    tmp = tmp[['left', 'right', 'height']]
    return tmp


def _make_timeseries_plot(source, figargs):
    tools = ['box_zoom', 'save', 'reset']
    if figargs['hover']:
        hover = HoverTool(
            tooltips=[
                ("Height", " @height"),
                ("Bin", " [@left{%R %F}, @right{%R %F})")
            ],
            formatters={
                'left': 'datetime',
                'right': 'datetime'
            },
            mode='mouse')
        tools += [hover]
    plot = figure(tools=tools, x_axis_type='datetime')
    if figargs['colors'] is None:
        plot_color = '#1F77B4'
        line_color = 'white'
    else:
        assert len(figargs['colors']) >= 2
        plot_color = figargs['colors'][0]
        line_color = figargs['colors'][1]

    plot.quad(top='height', bottom=0,
              left='left', right='right', color=plot_color,
              line_color=line_color, source=source, fill_alpha=.5)
    return plot


def _timeseries_widgets(col_1, col_2, col_max, col_min, n_bins, aggregate, callback):
    col_1_time = pd.to_datetime(col_1)
    if col_max is None:
        col_max = col_1_time.max()
    if col_min is None:
        col_min = col_1_time.min()

    slider = Slider(start=1, end=100,
                    value=n_bins, step=1,
                    title="Bins")
    slider.on_change('value', callback)

    range_select = DateRangeSlider(start=col_1_time.min(),
                                   end=col_1_time.max(),
                                   value=(col_min,
                                          col_max),
                                   step=1, title='Range', format='%R %F')
    range_select.on_change('value', callback)
    dropdown = Dropdown(value=aggregate, label=aggregate,
                        button_type="default",
                        menu=[('mean', 'mean'),
                              ('count', 'count'),
                              ('sum', 'sum'),
                              ('max', 'max'),
                              ('min', 'min')])
    dropdown.on_change('value', callback)
    return slider, range_select, dropdown

# Histogram Utilities #


def _make_histogram_source(col, y, n_bins, col_max, col_min, normalized):
    if col_max is None:
        col_max = col.max()
    if col_min is None:
        col_min = col.min()
    truncated = col[(col <= col_max) & (col >= col_min)]
    hist, edges = np.histogram(truncated, bins=n_bins, density=normalized)
    if normalized:
        hist = [height * (edges[1] - edges[0]) for height in hist]
    cols = pd.DataFrame({'col': col, 'label': y})
    tmp = pd.DataFrame({'hist': hist,
                        'left': edges[:-1],
                        'right': edges[1:]})
    if y is not None:
        label_hist = np.nan_to_num(cols['label'].groupby(
            pd.cut(col, edges, right=False)).sum().values)
        if normalized:
            label_hist = label_hist / (label_hist.sum())

        tmp['label'] = label_hist
    return tmp


def _make_histogram_plot(y, source, figargs):
    tools = ['box_zoom', 'save', 'reset']
    if figargs['hover']:
        if y is not None:
            hover = HoverTool(
                tooltips=[
                    ("Height", " @hist"),
                    ("Label", " @label"),
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
        tools += [hover]
    if figargs['colors'] is None:
        plot_1_color = '#1F77B4'
        plot_2_color = 'purple'
        line_color = 'white'
    else:
        assert len(figargs['colors']) >= 3
        plot_1_color = figargs['colors'][0]
        plot_2_color = figargs['colors'][1]
        line_color = figargs['colors'][2]

    plot = figure(tools=tools)
    plot.quad(top='hist', bottom=0, left='left',
              right='right', color=plot_1_color, line_color=line_color,
              source=source, fill_alpha=.5)

    if y is not None:
        plot.quad(top='label', bottom=0, left='left',
                  right='right', color=plot_2_color,
                  line_color=line_color, source=source, fill_alpha=.5)
    return plot


def _histogram_widgets(col, y, n_bins, col_max, col_min, callback):
    if col_max is None:
        col_max = col.max()
    if col_min is None:
        col_min = col.min()

    slider = Slider(start=1, end=100, value=n_bins, step=1, title="Bins")
    slider.on_change('value', callback)

    range_select = RangeSlider(start=col.min(),
                               end=col.max(),
                               value=(col_min, col_max),
                               step=5, title='Histogram Range')
    range_select.on_change('value', callback)
    return slider, range_select


# Scatter Utilities #

def _make_scatter_source(col_1, col_2, cat=None, label=None, aggregate='last'):
    tmp = pd.DataFrame({'col_1': col_1, 'col_2': col_2})

    if label is not None:
        tmp['label'] = label

    if cat is not None:
        tmp['cat'] = cat
        tmp = tmp.groupby('cat').aggregate(aggregate).reset_index()

    return tmp


def _make_scatter_plot(col_1, col_2, label, cat, source, figargs):
    tools = ['box_zoom', 'save', 'reset']
    if figargs['hover']:
        hover = HoverTool(tooltips=[
            (col_1.name, ' @col_1'),
            (col_2.name, ' @col_2'),
        ])
        if label is not None:
            hover.tooltips += [('label', ' @label')]

        if cat is not None:
            hover.tooltips += [('cat', ' @cat')]

        tools += [hover]
    radius = (col_1.max() - col_1.min()) / 100.
    plot = figure(tools=tools)
    if figargs['colors'] is not None:
        scatter_color = figargs['colors'][0]
    else:
        scatter_color = '#1F77B4'
    plot.scatter(x='col_1',
                 y='col_2',
                 color=scatter_color,
                 radius=radius,
                 source=source,
                 alpha=.8)

    return plot


def _scatter_widgets(col_1, col_2, aggregate, callback):
    dropdown = Dropdown(value=aggregate, label=aggregate,
                        button_type="default",
                        menu=[('mean', 'mean'),
                              ('last', 'last'),
                              ('sum', 'sum'),
                              ('max', 'max'),
                              ('min', 'min')])
    dropdown.on_change('value', callback)
    return dropdown
