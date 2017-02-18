from __future__ import absolute_import, division, print_function, unicode_literals
import six
import warnings
import types
from scikitplot.plotters import plot_silhouette, plot_elbow_curve


def clustering_factory(clf):
    """Takes a scikit-learn clusterer and embeds scikit-plot plotting methods in it.

    Args:
        clf: Scikit-learn clusterer instance

    Returns:
        The same scikit-learn clusterer instance passed in **clf** with embedded scikit-plot
        instance methods.

    Raises:
        ValueError: If **clf** does not contain the instance methods necessary for scikit-plot
            instance methods.
    """
    required_methods = ['fit', 'fit_predict']

    for method in required_methods:
        if not hasattr(clf, method):
            raise TypeError('"{}" is not in clf. Did you pass a clusterer instance?'.format(method))

    additional_methods = {
        'plot_silhouette': plot_silhouette,
        'plot_elbow_curve': plot_elbow_curve
    }

    for key, fn in six.iteritems(additional_methods):
        if hasattr(clf, key):
            warnings.warn('"{}" method already in clf. '
                          'Overriding anyway. This may result in unintended behavior.'.format(key))
        setattr(clf, key, types.MethodType(fn, clf))
    return clf
