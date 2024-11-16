.. currentmodule:: scikitplot.api.estimators

.. _estimators:

Estimators
======================================================================

This module contains functions related to estimators.


.. _regressor_model:

Regressor Model
----------------------------------------------------------------------

This module contains functions related to ``Regressor models``.


.. _classifier_model:

Classifier Model
----------------------------------------------------------------------

This module contains functions related to ``Classifier models``.


.. _plot_feature_importances:

plot feature importances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.estimators.plot_feature_importances`

Trained model of :class:`~sklearn.linear_model.LogisticRegression` and
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_classification_plot_feature_importances_script.py`: Example usage of
  :class:`~sklearn.linear_model.LogisticRegression` using the iris dataset

* :ref:`sphx_glr_auto_examples_classification_plot_feature_importances_script.py`: Example usage of
  :class:`~sklearn.ensemble.RandomForestClassifier` using the iris dataset


.. dropdown:: References

  * `"scikit-learn LogisticRegression coef_"
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#logisticregression>`_.

  * `"scikit-learn RandomForestClassifier feature_importances_"
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_>`_.


.. _plot_learning_curve:

plot_learning_curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.estimators.plot_learning_curve`

Trained model of :class:`~sklearn.linear_model.LogisticRegression` and
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_classification_plot_learning_curve_script.py`: Example usage of
  :class:`~sklearn.linear_model.LogisticRegression` using the iris dataset

* :ref:`sphx_glr_auto_examples_classification_plot_learning_curve_script.py`: Example usage of
  :class:`~sklearn.ensemble.RandomForestClassifier` using the iris dataset


.. dropdown:: References

  * `"scikit-learn learning_curve"
    <https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve>`_.


.. _cluster_model:

Cluster Model
----------------------------------------------------------------------

This module contains functions related to ``Cluster models``.


.. _plot_elbow:

plot elbow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.estimators.plot_elbow`

Trained model of :class:`~sklearn.cluster.KMeans` or :class:`~sklearn.cluster.MiniBatchKMeans`.
For an example of performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_clustering_plot_elbow_script.py`: Example usage of
  :class:`~sklearn.cluster.KMeans` using the iris dataset

.. dropdown:: References

  * `"scikit-learn k-means"
    <https://scikit-learn.org/stable/modules/clustering.html#k-means>`_.

  * `"scikit-learn mini-batch-k-means"
    <https://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means>`_.