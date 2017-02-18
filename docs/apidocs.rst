.. apidocs file containing the API Documentation
.. _factoryapidocs:

Factory API Reference
=====================

This document contains the plotting methods that are embedded into scikit-learn objects by the factory functions :func:`~scikitplot.clustering_factory`  and :func:`~scikitplot.classifier_factory`.

.. admonition:: Important Note

   If you want to use stand-alone functions and not bother with the factory functions, view the :ref:`functionsapidocs` instead.

Classifier Plots
----------------

.. autofunction:: scikitplot.classifier_factory

.. automodule:: scikitplot.classifiers
   :members: plot_learning_curve, plot_confusion_matrix, plot_roc_curve, plot_ks_statistic, plot_precision_recall_curve, plot_feature_importances

Clustering Plots
----------------

.. autofunction:: scikitplot.clustering_factory

.. automodule:: scikitplot.clustering
   :members: plot_silhouette, plot_elbow_curve