from __future__ import absolute_import, division, print_function, \
    unicode_literals
import six
import warnings
import types

from sklearn.utils import deprecated

from scikitplot.plotters import plot_silhouette, plot_elbow_curve


@deprecated('This will be removed in v0.4.0. The Factory '
            'API has been deprecated. Please migrate '
            'existing code into the various new modules '
            'of the Functions API. Please note that the '
            'interface of those functions will likely be '
            'different from that of the Factory API.')
def clustering_factory(clf):
    """Embeds scikit-plot plotting methods in an sklearn clusterer instance.

    Args:
        clf: Scikit-learn clusterer instance

    Returns:
        The same scikit-learn clusterer instance passed in **clf** with
        embedded scikit-plot instance methods.

    Raises:
        ValueError: If **clf** does not contain the instance methods necessary
            for scikit-plot instance methods.
    """
    required_methods = ['fit', 'fit_predict']

    for method in required_methods:
        if not hasattr(clf, method):
            raise TypeError('"{}" is not in clf. Did you '
                            'pass a clusterer instance?'.format(method))

    additional_methods = {
        'plot_silhouette': plot_silhouette,
        'plot_elbow_curve': plot_elbow_curve
    }

    for key, fn in six.iteritems(additional_methods):
        if hasattr(clf, key):
            warnings.warn('"{}" method already in clf. '
                          'Overriding anyway. This may '
                          'result in unintended behavior.'.format(key))
        setattr(clf, key, types.MethodType(fn, clf))
    return clf
