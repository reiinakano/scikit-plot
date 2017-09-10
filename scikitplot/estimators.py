"""
The :mod:`scikitplot.estimators` module includes plots built specifically for
scikit-learn estimator (classifier/regressor) instances e.g. Random Forest.
You can use your own estimators, but these plots assume specific properties
shared by scikit-learn estimators. The specific requirements are documented per
function.
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import warnings
import itertools

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import learning_curve
from sklearn.base import clone
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.utils import deprecated

from scipy import interp

from scikitplot.helpers import binary_ks_curve, validate_labels


def plot_feature_importances(clf, title='Feature Importance',
                             feature_names=None, max_num_features=20,
                             order='descending', x_tick_rotation=0, ax=None,
                             figsize=None, title_fontsize="large",
                             text_fontsize="medium"):
    """Generates a plot of a classifier's feature importances.

    Args:
        clf: Classifier instance that has a ``feature_importances_`` attribute,
            e.g. :class:`sklearn.ensemble.RandomForestClassifier` or
            :class:`xgboost.XGBClassifier`.

        title (string, optional): Title of the generated plot. Defaults to
            "Feature importances".

        feature_names (None, :obj:`list` of string, optional): Determines the
            feature names used to plot the feature importances. If None,
            feature names will be numbered.

        max_num_features (int): Determines the maximum number of features to
            plot. Defaults to 20.

        order ('ascending', 'descending', or None, optional): Determines the
            order in which the feature importances are plotted. Defaults to
            'descending'.

        x_tick_rotation (int, optional): Rotates x-axis tick labels by the
            specified angle. This is useful in cases where there are numerous
            categories and the labels overlap each other.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> rf = RandomForestClassifier()
        >>> rf.fit(X, y)
        >>> skplt.estimators.plot_feature_importances(
        ...     rf, feature_names=['petal length', 'petal width',
        ...                        'sepal length', 'sepal width'])
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_feature_importances.png
           :align: center
           :alt: Feature Importances
    """
    if not hasattr(clf, 'feature_importances_'):
        raise TypeError('"feature_importances_" attribute not in classifier. '
                        'Cannot plot feature importances.')

    importances = clf.feature_importances_

    if hasattr(clf, 'estimators_')\
            and isinstance(clf.estimators_, list)\
            and hasattr(clf.estimators_[0], 'feature_importances_'):
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)

    else:
        std = None

    if order == 'descending':
        indices = np.argsort(importances)[::-1]

    elif order == 'ascending':
        indices = np.argsort(importances)

    elif order is None:
        indices = np.array(range(len(importances)))

    else:
        raise ValueError('Invalid argument {} for "order"'.format(order))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if feature_names is None:
        feature_names = indices
    else:
        feature_names = np.array(feature_names)[indices]

    max_num_features = min(max_num_features, len(importances))

    ax.set_title(title, fontsize=title_fontsize)

    if std is not None:
        ax.bar(range(max_num_features),
               importances[indices][:max_num_features], color='r',
               yerr=std[indices][:max_num_features], align='center')
    else:
        ax.bar(range(max_num_features),
               importances[indices][:max_num_features],
               color='r', align='center')

    ax.set_xticks(range(max_num_features))
    ax.set_xticklabels(feature_names[:max_num_features],
                       rotation=x_tick_rotation)
    ax.set_xlim([-1, max_num_features])
    ax.tick_params(labelsize=text_fontsize)
    return ax
