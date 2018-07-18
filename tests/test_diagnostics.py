#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `notebook_utilities` package."""
import pandas as pd
import pytest

import henchman.diagnostics as diagnostics


@pytest.fixture
def df():
    """See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return pd.read_csv('./docs/fm.csv')


def test_overview(df, capsys):
    diagnostics.overview(df)
    output, _ = capsys.readouterr()
    split_output = output.split('\n')
    assert len(output) == 504
    assert len(split_output) == 29

    # Number of columns
    assert split_output[4][-3:] == u'117'
    # Number of rows
    assert split_output[5][-5:] == u'56353'

    # Average memory
    assert split_output[17][-7:] == u'0.61 MB'
    # Average missing
    assert split_output[11][-7:] == u'5569.66'

    # Bools
    assert split_output[24][-1:] == u'1'
    # Ints
    assert split_output[25][-2:] == u'36'
    # Objects
    assert split_output[27][-1:] == u'8'


def test_warnings(df, capsys):
    diagnostics.warnings(df)
    output, _ = capsys.readouterr()
    split_output = output.split('\n')
    assert len(output) == 9502
    assert len(split_output) == 109

    # distance group and average distance correlation
    assert split_output[50][-5:] == u'0.979'

    # missing values in skew delay
    assert split_output[102][-37:][:5] == u'56353'

    # unique airports
    assert split_output[107][-2:] == u'81'


def test_profile(df, capsys):
    diagnostics.profile(df.iloc[:, :15])
    output, _ = capsys.readouterr()
    split_output = output.split('\n')
    assert len(output) == 2775
    assert len(split_output) == 110

    # Number of columns
    assert split_output[4][-2:] == u'15'

    # Correlation of weekday of arrival and departure
    assert split_output[33][-5:] == u'0.912'

    # Catch all column types
    assert split_output[39] == u'|  Object Column Summary  |'
    assert split_output[55] == u'|  Boolean Column Summary  |'
    assert split_output[63] == u'|  Numeric Column Summary  |'

    # Mean of weekday time index
    assert split_output[91][-4:] == u'2.98'

    # Quartile 1 of flight date day
    assert split_output[104][-4:] == u'8.00'
