"""
Visualizations for model's performance-score metrics.

The :py:mod:`~scikitplot.api.metrics` module includes plots for machine learning evaluation metrics
e.g. confusion matrix, silhouette scores, etc.
"""
# scikitplot/api/metrics/__init__.py

# Your package/module initialization code goes here
from ._regression import *
from ._classification import *
from ._clustering import *

# Deprecated namespaces, to be removed in v0.5.0
from . import plotters