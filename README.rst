.. image:: https://badge.fury.io/py/fl-henchman.svg?maxAge=2592000
    :target: https://badge.fury.io/py/fl-henchman
.. image:: https://img.shields.io/conda/v/srothschild/henchman.svg
    :target: https://anaconda.org/srothschild/henchman
.. image:: https://img.shields.io/conda/pn/srothschild/henchman.svg 
    :target: https://anaconda.org/srothschild/henchman


Welcome to Henchman!
=====================
Henchman is a collection of `open source
<LICENSE>`_ python
utility functions for working in a jupyter notebook. With
Henchman, you can rapidly prototype end-to-end data science
workflows. You can explore data with
``henchman.diagnostics``, make interesting plots with
``henchman.plotting``, and do feature selection and machine
learning with ``henchman.selection`` and 
``henchman.learning``. 

For more information, visit the Henchman `documentation <https://henchman.featurelabs.com>`_.

Why?
~~~~~~~
Life is full of reusable functions. Here's what separates
Henchman:

- **Easy Interactive Plotting**: We bypass the flexible Bokeh
  API in favor of a small, rigid collection of standard data
  analysis plots. With sliders and checkboxes, finding the
  right plot parameters is as easy as a single function call.

.. image:: http://henchman.featurelabs.com/_images/piechart.gif
   :width: 47%
   :height: 300px
.. image:: http://henchman.featurelabs.com/_images/histogram.gif
   :width: 47%
   :height: 300px

- **Memorable API, Extensive documentation**: We have a
  heavy emphasis on ease of use. That means all the
  functions are sorted into one of 4 semantically named
  modules and names should be easy to remember inside that
  module. On top of that, every function has a docstring, an
  example and a `documentation <https://henchman.featurelabs.com>`_
  page.

.. image:: http://henchman.featurelabs.com/_images/create_model_docs.png
   :width: 75%
   :align: center

- **Novel Functionality**: We provide a few functions built
  from scratch to add to your data science workflow. There
  are methods to systematically find dataset attributes with
  ``overview`` and ``warnings`` from `henchman.diagnostics` and classes to
  select features in novel ways with ``RandomSelect`` and
  ``Dendrogram`` in `henchman.selection`.

.. image:: http://henchman.featurelabs.com/_images/overview.png
   :width: 47%
   :height: 300px
.. image:: http://henchman.featurelabs.com/_images/warnings.png
   :width: 47%
   :height: 300px

.. image:: http://henchman.featurelabs.com/_images/dendrogram.gif
   :align: center



Install
~~~~~~~~~
To install Henchman, run this command in your terminal:

.. code-block:: console

    $ python -m pip install fl-henchman

If you are using conda, you can download the most recent build from our channel on Anaconda.org:

.. code-block:: console

    $ conda install -c featurelabs henchman

These are the preferred methods to install Henchman, as it will always install the most recent stable release. If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

The sources for Henchman can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/featurelabs/henchman

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/featurelabs/henchman/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/featurelabs/henchman
.. _tarball: https://github.com/featurelabs/henchman/tarball/master








