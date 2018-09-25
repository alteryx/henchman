.. _api_ref:

API Reference
===============

Diagnostics API
~~~~~~~~~~~~~~~~

.. currentmodule:: henchman.diagnostics

.. autosummary::
    :toctree: generated/

    overview
    warnings
    column_report
    profile


Selection API
~~~~~~~~~~~~~~

.. currentmodule:: henchman.selection


.. autosummary::
    :toctree: generated/

    RandomSelect
    Dendrogram
    Dendrogram.fit
    Dendrogram.transform
    Dendrogram.set_params
    Dendrogram.features_at_step
    Dendrogram.find_set_of_size
    Dendrogram.score_at_point
    Dendrogram.shuffle_score_at_point

Learning API
~~~~~~~~~~~~~~

.. currentmodule:: henchman.learning


.. autosummary::
    :toctree: generated/

    create_model
    inplace_encoder
    feature_importances
    create_holdout

Plotting API
~~~~~~~~~~~~~

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

