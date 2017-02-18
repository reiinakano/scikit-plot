"""
This module contains a more flexible API for Scikit-plot users, exposing
simple functions to generate plots.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import learning_curve
from scipy import interp
import itertools
from scikitplot.helpers import binary_ks_curve
from sklearn.base import clone
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import cdist, pdist


def plot_confusion_matrix(y_true, y_pred, title=None, normalize=False, ax=None):
    """Generates confusion matrix plot for a given set of ground truth labels and classifier predictions.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_pred (array-like, shape (n_samples)):
            Estimated targets as returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to "Confusion Matrix" if
            `normalize` is True. Else, defaults to "Normalized Confusion Matrix.

        normalize (bool, optional): If True, normalizes the confusion matrix before plotting.
            Defaults to False.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
        >>> rf = RandomForestClassifier()
        >>> rf = rf.fit(X_train, y_train)
        >>> y_pred = rf.predict(X_test)
        >>> plot_confusion_matrix(y_test, y_pred, normalize=True)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_confusion_matrix.png
           :align: center
           :alt: Confusion matrix
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)

    if title:
        ax.set_title(title)
    elif normalize:
        ax.set_title('Normalized Confusion Matrix')
    else:
        ax.set_title('Confusion Matrix')

    image = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(mappable=image)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return ax


def plot_roc_curve(y_true, y_probas, title='ROC Curves', ax=None):
    """Generates the ROC curves for a set of ground truth labels and classifier probability predictions.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to "ROC Curves".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> plot_roc_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    classes = np.unique(y_true)
    probas = y_probas

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true, probas[:, i], pos_label=classes[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in fpr:
        i += 1
        micro_key += str(i)

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    fpr[micro_key], tpr[micro_key], _ = roc_curve(y_true.ravel(), probas.ravel())
    roc_auc[micro_key] = auc(fpr[micro_key], tpr[micro_key])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    macro_key = 'macro'
    i = 0
    while macro_key in fpr:
        i += 1
        macro_key += str(i)
    fpr[macro_key] = all_fpr
    tpr[macro_key] = mean_tpr
    roc_auc[macro_key] = auc(fpr[macro_key], tpr[macro_key])

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title(title)

    for i in range(len(classes)):
        ax.plot(fpr[i], tpr[i], lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(classes[i], roc_auc[i]))

    ax.plot(fpr[micro_key], tpr[micro_key],
            label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc[micro_key]),
            color='deeppink', linestyle=':', linewidth=4)
    ax.plot(fpr[macro_key], tpr[macro_key],
            label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc[macro_key]),
            color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    return ax


def plot_ks_statistic(y_true, y_probas, title='KS Statistic Plot', ax=None):
    """Generates the KS Statistic plot for a set of ground truth labels and classifier probability predictions.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to "KS Statistic Plot".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> plot_ks_statistic(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_ks_statistic.png
           :align: center
           :alt: KS Statistic
    """
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate KS statistic for data with '
                         '{} category/ies'.format(len(classes)))
    probas = y_probas

    # Compute KS Statistic curves
    thresholds, pct1, pct2, ks_statistic, \
        max_distance_at, classes = binary_ks_curve(y_true, probas[:, 1].ravel())

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title(title)

    ax.plot(thresholds, pct1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(thresholds, pct2, lw=3, label='Class {}'.format(classes[1]))
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(max_distance_at, *sorted([pct1[idx], pct2[idx]]),
               label='KS Statistic: {:.3f} at {:.3f}'.format(ks_statistic, max_distance_at),
               linestyle=':', lw=3, color='black')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Percentage below threshold')
    ax.legend(loc='lower right')

    return ax


def plot_precision_recall_curve(y_true, y_probas, title='Precision-Recall Curve', ax=None):
    """Generates the Precision Recall Curve for a set of ground truth labels and classifier probability predictions.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> plot_precision_recall_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    classes = np.unique(y_true)
    probas = y_probas

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_true, probas[:, i],
                                                            pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i], probas[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(y_true.ravel(),
                                                                        probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas, average='micro')

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title(title)
    for i in range(len(classes)):
        ax.plot(recall[i], precision[i], lw=2,
                label='Precision-recall curve of class {0} '
                      '(area = {1:0.3f})'.format(classes[i], average_precision[i]))
    ax.plot(recall[micro_key], precision[micro_key], lw=2, color='gold',
            label='micro-average Precision-recall curve '
                  '(area = {0:0.3f})'.format(average_precision[micro_key]))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='best')
    return ax


def plot_feature_importances(clf, title='Feature Importance', feature_names=None,
                             max_num_features=20, order='descending', ax=None):
    """Generates a plot of a classifier's feature importances.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict_proba`` methods.
            The classifier must also have a ``feature_importances_`` attribute.

        title (string, optional): Title of the generated plot. Defaults to "Feature importances".

        feature_names (None, :obj:`list` of string, optional): Determines the feature names used
            to plot the feature importances. If None, feature names will be numbered.

        max_num_features (int): Determines the maximum number of features to plot. Defaults to 20.

        order ('ascending', 'descending', or None, optional): Determines the order in which the
            feature importances are plotted. Defaults to 'descending'.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
            >>> rf = RandomForestClassifier()
            >>> rf.fit(X, y)
            >>> plot_feature_importances(rf, feature_names=['petal length', 'petal width',
            ...                                             'sepal length', 'sepal width'])
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
        fig, ax = plt.subplots(1, 1)

    if feature_names is None:
        feature_names = indices
    else:
        feature_names = np.array(feature_names)[indices]

    max_num_features = min(max_num_features, len(importances))

    ax.set_title(title)

    if std is not None:
        ax.bar(range(max_num_features), importances[indices][:max_num_features], color='r',
               yerr=std[indices][:max_num_features], align='center')
    else:
        ax.bar(range(max_num_features), importances[indices][:max_num_features],
               color='r', align='center')

    ax.set_xticks(range(max_num_features))
    ax.set_xticklabels(feature_names[:max_num_features])
    ax.set_xlim([-1, max_num_features])
    return ax


def plot_learning_curve(clf, X, y, title='Learning Curve', cv=None, train_sizes=None, n_jobs=1,
                        ax=None):
    """Generates a plot of the train and test learning curves for a given classifier.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict`` methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification or regression;
            None for unsupervised learning.

        title (string, optional): Title of the generated plot. Defaults to "Learning Curve"

        cv (int, cross-validation generator, iterable, optional): Determines the
            cross-validation strategy to be used for splitting.

            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        train_sizes (iterable, optional): Determines the training sizes used to plot the
            learning curve. If None, ``np.linspace(.1, 1.0, 5)`` is used.

        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
        >>> rf = classifier_factory(RandomForestClassifier())
        >>> rf.plot_learning_curve(X, y)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_learning_curve.png
           :align: center
           :alt: Learning Curve
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    ax.legend(loc="best")

    return ax


def plot_silhouette(clf, X, title='Silhouette Analysis', metric='euclidean', copy=True, ax=None):
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
            >>> kmeans = KMeans(n_clusters=4, random_state=1)
            >>> plot_silhouette(kmeans, X)
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


def plot_elbow_curve(clf, X, title='Elbow Plot', cluster_ranges=None, ax=None):
    """Plots elbow curve of different values of K for KMeans clustering.

    Args:
        clf: Clusterer instance that implements ``fit`` and ``fit_predict`` methods and an
            ``n_clusters`` parameter.

        X (array-like, shape (n_samples, n_features)):
            Data to cluster, where n_samples is the number of samples and
            n_features is the number of features.

        title (string, optional): Title of the generated plot. Defaults to "Elbow Plot"

        cluster_ranges (None or :obj:`list` of int, optional): List of n_clusters for which
            to plot the explained variances. Defaults to ``range(1, 12, 2)``.

        copy (boolean, optional): Determines whether ``fit`` is used on **clf** or on a
            copy of **clf**.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
            >>> kmeans = KMeans(random_state=1)
            >>> plot_elbow_curve(kmeans, cluster_ranges=range(1, 11))
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

    clfs = []
    for i in cluster_ranges:
        current_clf = clone(clf)
        setattr(current_clf, "n_clusters", i)
        clfs.append(current_clf.fit(X))

    centroids = [k.cluster_centers_ for k in clfs]

    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    dist = [np.min(D, axis=1) for D in D_k]
    # avgWithinSS = [np.sum(d)/X.shape[0] for d in dist]

    wcss = [np.sum(d**2) for d in dist]
    tss = np.sum(pdist(X)**2)/X.shape[0]
    bss = tss - wcss

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title(title)
    ax.plot(cluster_ranges, bss/tss*100, 'b*-')
    ax.grid(True)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Percent variance explained')

    return ax
