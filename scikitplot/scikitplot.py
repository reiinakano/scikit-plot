from __future__ import absolute_import
import warnings
import matplotlib.pyplot as plt


def classifier_factory(clf):
    """Takes a scikit-learn classifier instance and embeds scikit-plot instance methods in it.

    Args:
        clf: Scikit-learn classifier instance

    Returns:
        plot_clf: The scikit-learn classifier with embedded scikit-plot instance methods.

    Raises:
        ValueError: If `clf` does not contain the instance methods necessary for scikit-plot
        instance methods.
    """
    required_methods = ['fit', 'score', 'predict']

    for method in required_methods:
        if not hasattr(clf, method):
            raise ValueError('"{}" is not in clf. Did you pass a classifier instance?'.format(method))

    optional_methods = ['predict_proba']

    for method in optional_methods:
        if not hasattr(clf, method):
            warnings.warn('{} not in clf. Some plots may not be possible to generate.'.format(method))
