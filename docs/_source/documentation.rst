.. _documentation:

The ``Henchman`` package has reusable functionality in four areas: 
`dataframe diagnostics <#diagnostics>`_, `feature selection
<#selection>`_, `machine learning <#learning>`_ and `bokeh plotting <#plotting>`_.
We'll demonstrate some of that functionality here in the documentaion.

To get started, we'll use a premade feature matrix
from the Featuretools ``ft.demo.load_flight()`` function.
We'll load in the csv using pandas.

.. ipython:: python

    sample_df = pd.read_csv('../tests/sample_data/sample_fm.csv', index_col='trip_log_id').iloc[:100,:15]
    fm_enc = pd.read_csv('../tests/sample_data/sample_fm_enc.csv', index_col='trip_log_id')
  


.. currentmodule:: henchman

.. _diagnostics:

Diagnostics
============
It can sometimes be hard to find information about a dataframe by inspection.
Frequent questions such as "how large is this dataframe" and "are there duplicates"
usually require copying code from one notebook to another. In this module, we give easy
function calls that do those basic diagnostics.

.. ipython:: python

    from henchman.diagnostics import overview
    overview(fm_enc)


With just the :func:`overview <henchman.diagnostics.overview>` function call, we've answered many of our basic questions
about this dataframe. We know that it has 349 columns, 56353 rows and is about 162 MB.
Almost all of the columns are integers or floats in ``pandas``.

It's also useful to see warnings about common data science
pitfalls. The :func:`warnings <henchman.diagnostics.warnings>` function
checks the pairwise correlation of all of your columns, if you have any duplicates, if there are
many values missing from a column and if you have object columns with many distinct values. Let's
sample the warning function on a smaller dataframe.

.. ipython:: python

    from henchman.diagnostics import warnings
    warnings(sample_df)

It's not particularly suprising that the distance would be 
highly correlated with how long a flight will take!
Nevertheless, it's a good thing to know before feeding both
of those columns into a machine learning algorithm. It also
seems like we shouldn't be encoding the flight id, since it
would give too many unique values. A better approach (and
the one that was taken in this feature matrix) is to 
aggregate according to that id.

While technically the :func:`overview <diagnostics.overview>`, :func:`warnings <diagnostics.warnings>` and :func:`column_report <diagnostics.column_report>` are three 
distinct API-stable functions, the most common use case is to call all three at once using the
:func:`profile <henchman.diagnostics.profile>` function. This will do the same overview and warnings as above but also give information
about every column in the dataframe.

.. ipython:: python

    from henchman.diagnostics import profile
    profile(sample_df)

**Module Contents**

.. currentmodule:: henchman.diagnostics


.. autosummary::
    :toctree: generated/

    overview
    warnings
    column_report
    profile



.. _selection:

Selection
==========

There are some lightweight feature selection packages
provided as well. There is
:class:`RandomSelect <henchman.selection.RandomSelect>`
randomly choose ``n_feats`` features to select and
:class:`Dendrogram <henchman.selection.Dendrogram>`
which can use pairwise correlation to find a feature set.

.. ipython:: python

    from henchman.selection import RandomSelect
    X = fm_enc.copy().fillna(0).iloc[:100, :30]
    y = fm_enc.copy().pop('label')[:100]
    sel = RandomSelect(n_feats=12)
    sel.fit(X)
    sel.transform(X).head()

Alternately, you can use pairwise correlation

.. ipython:: python
    :okwarning:

    from henchman.selection import Dendrogram
    sel2 = Dendrogram(X)
    sel2.transform(X, n_feats=12).head()

The Dendrogram can not necessarily find a feature set of an 
arbitrary size. Since not all features are pairwise
correlated, not all features will eventually be connected.
Since the object has total connectivity information
according to the given metric, if there is a distance
that provides a better indicator of feature closeness in
your dataset it can be passed in as an argument.

As a last note, the features returned are actually
*representatives* of a connected component of a particular
graph. Those components can be seen, and the representatives
can be shuffled to return a similarly connected feature set.

.. ipython:: python

    sel2.graphs[4]
    sel2.shuffle_all_representatives()
    sel2.graphs[4]

**Module Contents**

.. currentmodule:: henchman.selection


.. autosummary::
    :toctree: generated/

    RandomSelect
    Dendrogram

.. _learning:

Learning
==========
The learning module exists to simplify some frequent machine
learning calls. For instance, given a feature matrix ``X``
and a column of labels ``y``, it's nice to be able to
quickly get back a score. We'll use the :func:`create_model
<henchman.learning.create_model>` function.

.. ipython:: python

    from henchman.learning import create_model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    scores, fit_model = create_model(X, y, RandomForestClassifier(), roc_auc_score)
    scores

We can check :func:`feature importances
<henchman.learning.feature_importances>` with similar ease:

.. ipython:: python

    from henchman.learning import feature_importances
    feats = feature_importances(X, fit_model, n_feats=3)
    X[feats].head()

**Module Contents**

.. currentmodule:: henchman.learning


.. autosummary::
    :toctree: generated/

    create_model
    inplace_encoder
    feature_importances
    create_holdout


.. _plotting:

Plotting 
=========

The plotting module gives a collection of useful dataset
agnostic plots. Plots have the ability to be *dynamic* or
*static*. We recommend importing the whole module at once
using ``import henchman.plotting as hplot`` for easy access
to all of the functions. The single exception might be
:func:`henchman.plotting.show`, which is useful enough that
you might consider importing it as itself.

The :func:`show <henchman.plotting.show>` function has many parameters which can be hard
to remember. Because of that, there's a :func:`templating function <henchman.plotting.show_template>` from which you can copy and paste the arguments you want.

.. ipython:: python

    import henchman.plotting as hplot
    from henchman.plotting import show
    hplot.show_template()

See the `plotting gallery <plotting_gallery.html>`_ page
for some example bokeh plots.

 **Module Contents**

.. currentmodule:: henchman.plotting


.. autosummary::
    :toctree: generated/

    show
    show_template
    piechart
    histogram
    timeseries
    scatter
    dendrogram
    feature_importances
    roc_auc
    f1
