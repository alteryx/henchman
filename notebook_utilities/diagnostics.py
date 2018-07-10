# -*- coding: utf-8 -*-

'''The diagnostics module. Describe a particular dataset.

Contents:
        overview (df): Describe the shape, NaNs and Memory usage
        warnings (df): Find dups, corrs and high dim categoricals
        column_report (df): Describe all columns according to pandas dtype
        profile (df): Do the above 3 functions.
'''
import numpy as np
import pandas as pd


def title(string):
    centerline = '|  {}  |'.format(string)
    border = '+' + '{}'.format('-' * (len(centerline) - 2)) + '+'
    print('')
    print(border)
    print(centerline)
    print(border)


def subtitle(string):
    print('')
    print('## {} ##'.format(string))


def overview(data):
    title('Data Shape')
    print('Number of columns: {}'.format(data.shape[1]))
    print('Number of rows: {}'.format(data.shape[0]))

    title('Missing Values')
    missing_values = data.isnull().sum().sort_values()
    print('Most values missing from column: {}'.format(missing_values[-1]))
    print('Average missing values by column: {:.2f}'.format(
        missing_values.mean()))

    title('Memory Usage')
    memory_used = data.memory_usage(deep=True)/1000000
    print('Total memory used: {:.2f} MB'.format(memory_used.sum()))
    print('Average memory by column: {:.2f} MB'.format(memory_used.mean()))

    title('Data Types')
    print(pd.DataFrame([data[col].dtype for col in data]
                       ).reset_index().groupby(0).count())


def _find_duplicates(data):
    duplicates = data[data.duplicated()]
    if duplicates.shape[0] > 0:
        print('DataFrame has {} duplicates'.format(duplicates.shape[0]))


def _find_correlations(data, corr_thresh):
    correlations = data.corr()
    warningfm = correlations[(np.abs(correlations) > corr_thresh) & (
        np.abs(correlations) < 1.)]
    listed = []
    unicorr = []
    for col in warningfm:
        warningcol = warningfm[col][~warningfm[col].isnull()]
        if not warningcol.empty:
            for index, value in warningcol.iteritems():
                if (index, col) not in listed:
                    print('{} and {} are linearly correlated: {:.3f}'.format(
                        col, index, value))
                    listed.append((col, index))
                    listed.append((index, col))
                    unicorr += [col, index]


def _find_missing(data, missing_thresh):
    for index, value in data.isnull().sum().iteritems():
        if value > (data.shape[0] * missing_thresh):
            print('{} has {} missing values: ({}% of total)'.format(
                index, value, 100 * value / data.shape[0]))


def _find_high_card(data, card_thresh):
    objects = [col for col in data if data[col].dtype == 'O']
    for index, value in data[objects].nunique().iteritems():
        if value > card_thresh:
            print('{} has many unique values: {}'.format(index, value))


def warnings(data, corr_thresh=.9, missing_thresh=.1, card_thresh=50):
    '''Warn about common dataset problems.
    Input:
        data (df): The dataframe to warn about.
        corr_thresh (float): Warn above this threshold (Default .9)
        missing_thresh (float): Warn above this threshold (Default .1)
        card_thresh (int): Warn above this threshold (Default 50).
    '''
    title('Warnings')
    _find_duplicates(data)
    _find_correlations(data, corr_thresh)
    _find_missing(data, missing_thresh)
    _find_high_card(data, card_thresh)


def _object_column_summary(data, objects):
    title('Object Column Summary')
    for col in objects:
        subtitle(col)
        datacol = data[col]
        print('Unique: {}'.format(len(datacol.unique())))

        mode = datacol.mode().values
        if len(mode) > 1:
            print('Mode: No Mode')

        else:
            mode = mode[0]
            nummode = 100 * datacol[datacol == mode].shape[0]/datacol.shape[0]
            print('Mode: {}, (matches {:.1f}% of rows)'.format(mode, nummode))
        missing = datacol.isnull().sum()
        if missing > 0:
            print('Missing: {}'.format(missing))


def _time_column_summary(data, times):
    title('Time Column Summary')
    for col in times:
        subtitle(col)
        datacol = data[col]
        print('Last Time: {}'.format(datacol.max()))
        print('First Time: {}'.format(datacol.min()))


def _boolean_column_summary(data, bools):
    title('Boolean Column Summary')
    for col in bools:
        subtitle(col)
        datacol = data[col]
        numtrue = float(datacol.sum())
        total = datacol.shape[0]
        perctrue = 100 * numtrue / total

        print('Number True: {}, Number False: {}, Mean: {:.2f}'.format(
            numtrue, total, datacol.mean()))
        print('Percent True: {:.1f}% | Percent False: {:.1f}%'.format(
            perctrue, 100 - perctrue))
        missing = datacol.isnull().sum()
        if missing > 0:
            print('Missing: {}'.format(missing))


def _numeric_column_summary(data, numbers):
    title('Numeric Column Summary')
    for col in numbers:
        subtitle(col)
        datacol = data[col]
        print('Maximum: {}, Minimum: {}, Mean: {:.2f}'.format(
            datacol.max(), datacol.min(), datacol.mean()))
        print('Quartile 3: {:.2f} | Median: {:.2f}'
              '| Quartile 1: {:.2f}'.format(datacol.quantile(.75),
                                            datacol.quantile(.5),
                                            datacol.quantile(.25)))
        missing = datacol.isnull().sum()
        if missing > 0:
            print('Missing: {}'.format(missing))


def column_report(data):
    objects = [col for col in data if data[col].dtype == 'O']
    times = [col for col in data if data[col].dtype == '<M8[ns]']
    bools = [col for col in data if data[col].dtype == 'bool']
    numbers = [col for col in data if data[col].dtype in (
        ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])]
    if objects != []:
        _object_column_summary(data, objects)
    if times != []:
        _time_column_summary(data, times)
    if bools != []:
        _boolean_column_summary(data, bools)
    if numbers != []:
        _numeric_column_summary(data, numbers)


def profile(data, corr_thresh=.9, missing_thresh=.1, card_thresh=50):
    '''Profile dataset.
    Input:
        data (df): The dataframe to profile.
        corr_thresh (float): Warn above this threshold (Default .9)
        missing_thresh (float): Warn above this threshold (Default .1)
        card_thresh (int): Warn above this threshold (Default 50).
    '''
    overview(data)
    warnings(data, corr_thresh, missing_thresh, card_thresh)
    column_report(data)
