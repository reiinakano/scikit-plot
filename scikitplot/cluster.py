"""
The :mod:`scikitplot.cluster` module includes plots built specifically for
scikit-learn clusterer instances e.g. KMeans. You can use your own clusterers,
but these plots assume specific properties shared by scikit-learn estimators.
The specific requirements are documented per function.
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import clone
from joblib import Parallel, delayed


def plot_elbow_curve(clf, X, title='Elbow Plot', cluster_ranges=None, n_jobs=1,
                     show_cluster_time=True, ax=None, figsize=None,
                     title_fontsize="large", text_fontsize="medium"):
    """Plots elbow curve of different values of K for KMeans clustering.

    Args:
        clf: Clusterer instance that implements ``fit``,``fit_predict``, and
            ``score`` methods, and an ``n_clusters`` hyperparameter.
            e.g. :class:`sklearn.cluster.KMeans` instance

        X (array-like, shape (n_samples, n_features)):
            Data to cluster, where n_samples is the number of samples and
            n_features is the number of features.

        title (string, optional): Title of the generated plot. Defaults to
            "Elbow Plot"

        cluster_ranges (None or :obj:`list` of int, optional): List of
            n_clusters for which to plot the explained variances. Defaults to
            ``range(1, 12, 2)``.

        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to
            1.

        show_cluster_time (bool, optional): Include plot of time it took to
            cluster for a particular K.

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
        >>> kmeans = KMeans(random_state=1)
        >>> skplt.cluster.plot_elbow_curve(kmeans, cluster_ranges=range(1, 30))
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_elbow_curve.png
           :align: center
           :alt: Elbow Curve
    """
    if cluster_ranges is None:
        cluster_ranges = range(1, 12, 2)
    else:
        cluster_ranges = sorted(cluster_ranges)

    if not hasattr(clf, 'n_clusters'):
        raise TypeError('"n_clusters" attribute not in classifier. '
                        'Cannot plot elbow method.')

    tuples = Parallel(n_jobs=n_jobs)(delayed(_clone_and_score_clusterer)
                                     (clf, X, i) for i in cluster_ranges)
    clfs, times = zip(*tuples)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.plot(cluster_ranges, np.absolute(clfs), 'b*-')
    ax.grid(True)
    ax.set_xlabel('Number of clusters', fontsize=text_fontsize)
    ax.set_ylabel('Sum of Squared Errors', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)

    if show_cluster_time:
        ax2_color = 'green'
        ax2 = ax.twinx()
        ax2.plot(cluster_ranges, times, ':', alpha=0.75, color=ax2_color)
        ax2.set_ylabel('Clustering duration (seconds)',
                       color=ax2_color, alpha=0.75,
                       fontsize=text_fontsize)
        ax2.tick_params(colors=ax2_color, labelsize=text_fontsize)

    return ax


def _clone_and_score_clusterer(clf, X, n_clusters):
    """Clones and scores clusterer instance.

    Args:
        clf: Clusterer instance that implements ``fit``,``fit_predict``, and
            ``score`` methods, and an ``n_clusters`` hyperparameter.
            e.g. :class:`sklearn.cluster.KMeans` instance

        X (array-like, shape (n_samples, n_features)):
            Data to cluster, where n_samples is the number of samples and
            n_features is the number of features.

        n_clusters (int): Number of clusters

    Returns:
        score: Score of clusters

        time: Number of seconds it took to fit cluster
    """
    start = time.time()
    clf = clone(clf)
    setattr(clf, 'n_clusters', n_clusters)
    return clf.fit(X).score(X), time.time() - start
