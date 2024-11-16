"""
Visualizations for matrix decomposition algorithms.

The :py:mod:`~scikitplot.api.decomposition` module includes plots built specifically for
scikit-learn estimators that are used for dimensionality reduction e.g. PCA.

You can use your own estimators, but these plots assume specific properties shared by
scikit-learn estimators. The specific requirements are documented per function.
"""
# scikitplot/api/decomposition/__init__.py

# Your package/module initialization code goes here
from ._components import *
from ._projection import *