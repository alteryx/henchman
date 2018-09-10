#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the diagnostics module."""
import pandas as pd
import pytest

import henchman.diagnostics as diagnostics


@pytest.fixture
def df():
    """See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return pd.read_csv('./tests/sample_data/sample_fm.csv')


def test_overview(df, capsys):
    diagnostics.overview(df)
    output, _ = capsys.readouterr()
    split_output = output.split('\n')

    assert len(split_output) == 29

    # Number of columns
    assert split_output[4][-3:] == u'110'
    # Number of rows
    assert split_output[5][-3:] == u'100'

    # Average memory
    # assert split_output[17][-7:] == u'0.61 MB'
    # Average missing
    assert split_output[11][-4:] == u'5.35'

    # Bools
    assert split_output[24][-1:] == u'1'
    # Ints
    assert split_output[25][-2:] == u'28'
    # Objects
    assert split_output[27][-1:] == u'9'


def test_warnings(df, capsys):
    diagnostics.warnings(df)
    output, _ = capsys.readouterr()
    split_output = output.split('\n')
    assert len(split_output) == 86

    # Max and min scheduled elapsed time corr
    assert split_output[50][-5:] == u'0.999'

    # Unique flight ids
    assert split_output[84][-2:] == u'98'


def test_profile(df, capsys):
    diagnostics.profile(df.iloc[:, :15])
    output, _ = capsys.readouterr()
    split_output = output.split('\n')

    assert len(split_output) == 104

    # Number of columns
    assert split_output[4][-2:] == u'15'

    # Correlation of day of arrival and departure
    assert split_output[33][-5:] == u'1.000'

    # Catch all column types
    assert split_output[37] == u'|  Object Column Summary  |'
    assert split_output[49] == u'|  Numeric Column Summary  |'

    # year scheduled_arr_time
    assert split_output[90][-7:] == u'2017.00'

    # Quartile 1 of scheduled month time
    assert split_output[102][-4:] == u'1.00'


def test_numeric_summary(df, capsys):
    diagnostics.column_report(df[['scheduled_elapsed_time']])
    output, _ = capsys.readouterr()
    split_output = output.split('\n')
    true_list = ['## scheduled_elapsed_time ##',
                 'Maximum: 24780000000000, Minimum: 3900000000000, Mean: 10066800000000.00',
                 'Quartile 3: 12990000000000.00 | Median: 8400000000000.00'
                 '| Quartile 1: 5490000000000.00',
                 '']
    for i, value in enumerate(split_output[5:]):
        assert value == true_list[i]


def test_boolean_summary(df, capsys):
    diagnostics.column_report(df[['label']])
    output, _ = capsys.readouterr()
    split_output = output.split('\n')
    true_list = ['## label ##',
                 'Number True: 11.0, Number False: 89.0, Mean: 0.11',
                 'Percent True: 11.0% | Percent False: 89.0%', '']
    for i, value in enumerate(split_output[5:]):
        assert value == true_list[i]


def test_object_summary(df, capsys):
    diagnostics.column_report(df[['flights.carrier']])
    output, _ = capsys.readouterr()
    split_output = output.split('\n')
    true_list = ['## flights.carrier ##', 'Unique: 10',
                 'Mode: WN, (matches 23.0% of rows)', '']
    for i, value in enumerate(split_output[5:]):
        assert value == true_list[i]


def test_time_summary(df, capsys):
    timetest = df.reset_index()[['time']]
    timetest['time'] = pd.to_datetime(timetest['time'])

    diagnostics.column_report(timetest)
    output, _ = capsys.readouterr()
    split_output = output.split('\n')

    true_list = ['## time ##',
                 'Last Time: 2017-02-25 00:00:00',
                 'First Time: 2016-12-29 00:00:00', '']
    for i, value in enumerate(split_output[5:]):
        assert value == true_list[i]
