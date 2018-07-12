Henchman
=============
A collection of utility functions for making demo notebooks.

* Free software: GNU General Public License v3
* Documentation: https://notebook-utilities.readthedocs.io.

To install the development version with pip: 
1. download the repository 
2. open a terminal and navigate to the downloaded folder
3. run ``pip install -e .``

Functionality
~~~~~~~~~~~~~

The Henchman package is a collection of frequently used utility functions for Featuretools demos. There are a number of functions which appear in multiple `utils.py` files in multiple demos. This project consolidates those into 4 categories of reusable functions.

Diagnostics
-----------
It's often useful to have an all-at-once overview of what is going on in a particular dataframe or feature matrix. The diagnostics module gives basic print functionality for many commonly asked questions. You can import the ``profile`` function with ``from henchman.diagnostics import profile``.

Selection
---------
Given the number of features that Featuretools is capable of creating, it's worthwhile having some easily usable feature selection techniques. This module provides a wrapper around some scikit-learn methods.

Learning
--------
We reuse the same workflow in multiple demos once we have transformed our feature matrix into an ``X`` and ``y``. The ``create_model`` function gives a basic validation split and scores according to a provided model and metric. To use, import with ``from henchman.learning import create_model``.

Plotting
--------
It's usually preferable to look at a plot than a dataframe, but it can take a lot of time to make a perfect looking plot. This module gives a simple interface for certain plots that we know we'll need often.

Credits
~~~~~~~

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
