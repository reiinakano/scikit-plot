from __future__ import absolute_import, division, print_function, unicode_literals
import six
import warnings
import types
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.base import clone
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


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
        'plot_silhouette': plot_silhouette
    }

    for key, fn in six.iteritems(additional_methods):
        if hasattr(clf, key):
            warnings.warn('"{}" method already in clf. '
                          'Overriding anyway. This may result in unintended behavior.'.format(key))
        setattr(clf, key, types.MethodType(fn, clf))
    return clf


def plot_silhouette(clf, X, title='Silhouette Analysis', metric='euclidean', copy=False, ax=None):
    """Plots silhouette analysis of clusters using fit_predict.

    Args:
        clf: Clusterer instance that implements ``fit`` and ``fit_predict`` methods.

        X (array-like, shape (n_samples, n_features)):
            Data to cluster, where n_samples is the number of samples and
            n_features is the number of features.

        title (string, optional): Title of the generated plot. Defaults to "Silhouette Analysis"

        metric (string or callable, optional): The metric to use when calculating distance
            between instances in a feature array. If metric is a string, it must be one of
            the options allowed by sklearn.metrics.pairwise.pairwise_distances. If X is
            the distance array itself, use "precomputed" as the metric.

        copy (boolean, optional): Determines whether ``fit`` is used on **clf** or on a
            copy of **clf**.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
            >>> kmeans = clustering_factory(KMeans(n_clusters=4, random_state=1))
            >>> kmeans.plot_silhouette(X)
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
            >>> plt.show()

        .. image:: _static/examples/plot_silhouette.png
           :align: center
           :alt: Silhouette Plot
    """
    if copy:
        clf = clone(clf)

    cluster_labels = clf.fit_predict(X)

    n_clusters = len(set(cluster_labels))

    silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)

    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title(title)
    ax.set_xlim([-0.1, 1])

    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10 + 10])

    ax.set_xlabel('Silhouette coefficient values')
    ax.set_ylabel('Cluster label')

    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label='Silhouette score: {0:0.3f}'.format(silhouette_avg))

    ax.set_yticks([])  # Clear the y-axis labels / ticks
    ax.set_xticks(np.arange(-0.1, 1.0, 0.2))

    ax.legend(loc='best')

    return ax
