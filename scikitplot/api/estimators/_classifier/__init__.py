"""
Classifier instances
============================================================================

The :py:mod:`~scikitplot.estimators` module includes plots built specifically for
scikit-learn estimator (classifier/regressor) instances e.g. Random Forest.

You can use your own estimators, but these plots assume specific properties shared by
scikit-learn estimators. The specific requirements are documented per function.
"""
# scikitplot/api/estimators/_classifier/__init__.py

# Your package/module initialization code goes here
from ._feature_importances import *
from ._learning_curve import *