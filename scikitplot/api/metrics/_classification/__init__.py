"""
Classification metrics
========================================================

The :py:mod:`~scikitplot.metrics` module includes plots for machine learning evaluation metrics
e.g. confusion matrix, silhouette scores, etc.
"""
# scikitplot/api/metrics/_classification/__init__.py

# Your package/module initialization code goes here
from ._calibration import (
  plot_calibration,
)
from ._confusion_matrix import (
  plot_confusion_matrix,
  plot_classifier_eval,
)
from ._precision_recall_curve import (
  plot_precision_recall_curve,
  plot_precision_recall,
)
from ._roc_curve import (
  plot_roc_curve,
  plot_roc,
)