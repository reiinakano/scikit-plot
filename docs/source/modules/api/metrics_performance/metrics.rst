.. currentmodule:: scikitplot.api.metrics

.. _metrics:

Metrics
======================================================================

This module contains functions related to metrics.


.. _regression_metrics:

Regression metrics
----------------------------------------------------------------------

This module contains functions related to ``Regression metrics``.

.. _plot_residuals_distribution:

plot residuals distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.metrics.plot_residuals_distribution`

Trained model of :class:`~sklearn.linear_model.LinearRegression` or
:class:`~sklearn.ensemble.RandomForestRegressor`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_regression_plot_residuals_distribution_script.py`: Example usage of
  :class:`sklearn.linear_model.LinearRegression` using the diabetes dataset (regression).


.. dropdown:: References

  * `"Normal Probability Plot of Residuals"
    <https://online.stat.psu.edu/stat462/node/122/>`_.



.. _classification_metrics:

Classification metrics
----------------------------------------------------------------------

This module contains functions related to ``Classification metrics``.


.. _plot_calibration:

plot calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.metrics.plot_calibration`

Trained model of :class:`~sklearn.linear_model.LogisticRegression` or
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_calibration_plot_calibration_script.py`: Example usage of
  :class:`sklearn.linear_model.LogisticRegression` using the iris dataset


.. dropdown:: References

  * `"scikit-learn PCA"
    <https://scikit-learn.org/stable/auto_examples/calibration/index.html#calibration>`_.

.. _plot_precision_recall:

plot precision recall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.metrics.plot_precision_recall`

Trained model of :class:`~sklearn.linear_model.LogisticRegression` or
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_classification_plot_precision_recall_script.py`: Example usage of
  :class:`~sklearn.linear_model.LogisticRegression` using the iris dataset

.. dropdown:: References

  * `"scikit-learn precision-recall"
    <https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#precision-recall>`_.


.. _plot_roc:

plot roc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.metrics.plot_roc`

Trained model of :class:`~sklearn.linear_model.LogisticRegression` or
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_classification_plot_roc_script.py`: Example usage of
  :class:`~sklearn.linear_model.LogisticRegression` using the iris dataset

.. dropdown:: References

  * `"scikit-learn roc"
    <https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html>`_.


.. _clustering_metrics:

Clustering metrics
----------------------------------------------------------------------

This module contains functions related to ``Clustering metrics``.


.. _plot_silhouette:

plot silhouette
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.metrics.plot_silhouette`

Trained model of :class:`~sklearn.cluster.KMeans` or :class:`~sklearn.cluster.MiniBatchKMeans`.
For an example of performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_clustering_plot_silhouette_script.py`: Example usage of
  :class:`~sklearn.cluster.KMeans` using the iris dataset

.. dropdown:: References

  * `"scikit-learn k-means"
    <https://scikit-learn.org/stable/modules/clustering.html#k-means>`_.

  * `"scikit-learn mini-batch-k-means"
    <https://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means>`_.