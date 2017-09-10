"""
The :mod:`scikitplot.decomposition` module includes plots built specifically
for scikit-learn estimators that are used for dimensionality reduction
e.g. PCA. You can use your own estimators, but these plots assume specific
properties shared by scikit-learn estimators. The specific requirements are
documented per function.
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


def plot_pca_component_variance(clf, title='PCA Component Explained Variances',
                                target_explained_variance=0.75, ax=None,
                                figsize=None, title_fontsize="large",
                                text_fontsize="medium"):
    """Plots PCA components' explained variance ratios. (new in v0.2.2)

    Args:
        clf: PCA instance that has the ``explained_variance_ratio_`` attribute.

        title (string, optional): Title of the generated plot. Defaults to
            "PCA Component Explained Variances"

        target_explained_variance (float, optional): Looks for the minimum
            number of principal components that satisfies this value and
            emphasizes it on the plot. Defaults to 0.75

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
        >>> pca = PCA(random_state=1)
        >>> pca.fit(X)
        >>> skplt.decomposition.plot_pca_component_variance(pca)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_pca_component_variance.png
           :align: center
           :alt: PCA Component variances
    """
    if not hasattr(clf, 'explained_variance_ratio_'):
        raise TypeError('"clf" does not have explained_variance_ratio_ '
                        'attribute. Has the PCA been fitted?')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    cumulative_sum_ratios = np.cumsum(clf.explained_variance_ratio_)

    # Magic code for figuring out closest value to target_explained_variance
    idx = np.searchsorted(cumulative_sum_ratios, target_explained_variance)

    ax.plot(range(len(clf.explained_variance_ratio_) + 1),
            np.concatenate(([0], np.cumsum(clf.explained_variance_ratio_))),
            '*-')
    ax.grid(True)
    ax.set_xlabel('First n principal components', fontsize=text_fontsize)
    ax.set_ylabel('Explained variance ratio of first n components',
                  fontsize=text_fontsize)
    ax.set_ylim([-0.02, 1.02])
    if idx < len(cumulative_sum_ratios):
        ax.plot(idx+1, cumulative_sum_ratios[idx], 'ro',
                label='{0:0.3f} Explained variance ratio for '
                'first {1} components'.format(cumulative_sum_ratios[idx],
                                              idx+1),
                markersize=4, markeredgewidth=4)
        ax.axhline(cumulative_sum_ratios[idx],
                   linestyle=':', lw=3, color='black')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="best", fontsize=text_fontsize)

    return ax
