"""
The :mod:`scikitplot.metrics` module includes plots for machine learning
evaluation metrics e.g. confusion matrix, silhouette scores, etc.
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


def plot_confusion_matrix(y_true, y_pred, labels=None, true_labels=None,
                          pred_labels=None, title=None, normalize=False,
                          hide_zeros=False, x_tick_rotation=0, ax=None,
                          figsize=None, cmap='Blues', title_fontsize="large",
                          text_fontsize="medium"):
    """Generates confusion matrix plot from predictions and true labels

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_pred (array-like, shape (n_samples)):
            Estimated targets as returned by a classifier.

        labels (array-like, shape (n_classes), optional): List of labels to
            index the matrix. This may be used to reorder or select a subset
            of labels. If none is given, those that appear at least once in
            ``y_true`` or ``y_pred`` are used in sorted order. (new in v0.2.5)

        true_labels (array-like, optional): The true labels to display.
            If none is given, then all of the labels are used.

        pred_labels (array-like, optional): The predicted labels to display.
            If none is given, then all of the labels are used.

        title (string, optional): Title of the generated plot. Defaults to
            "Confusion Matrix" if `normalize` is True. Else, defaults to
            "Normalized Confusion Matrix.

        normalize (bool, optional): If True, normalizes the confusion matrix
            before plotting. Defaults to False.

        hide_zeros (bool, optional): If True, does not plot cells containing a
            value of zero. Defaults to False.

        x_tick_rotation (int, optional): Rotates x-axis tick labels by the
            specified angle. This is useful in cases where there are numerous
            categories and the labels overlap each other.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

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
        >>> rf = rf.fit(X_train, y_train)
        >>> y_pred = rf.predict(X_test)
        >>> skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_confusion_matrix.png
           :align: center
           :alt: Confusion matrix
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        classes = unique_labels(y_true, y_pred)
    else:
        classes = np.asarray(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    if true_labels is None:
        true_classes = classes
    else:
        validate_labels(classes, true_labels, "true_labels")

        true_label_indexes = np.in1d(classes, true_labels)

        true_classes = classes[true_label_indexes]
        cm = cm[true_label_indexes]

    if pred_labels is None:
        pred_classes = classes
    else:
        validate_labels(classes, pred_labels, "pred_labels")

        pred_label_indexes = np.in1d(classes, pred_labels)

        pred_classes = classes[pred_label_indexes]
        cm = cm[:, pred_label_indexes]

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    elif normalize:
        ax.set_title('Normalized Confusion Matrix', fontsize=title_fontsize)
    else:
        ax.set_title('Confusion Matrix', fontsize=title_fontsize)

    image = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    plt.colorbar(mappable=image)
    x_tick_marks = np.arange(len(pred_classes))
    y_tick_marks = np.arange(len(true_classes))
    ax.set_xticks(x_tick_marks)
    ax.set_xticklabels(pred_classes, fontsize=text_fontsize,
                       rotation=x_tick_rotation)
    ax.set_yticks(y_tick_marks)
    ax.set_yticklabels(true_classes, fontsize=text_fontsize)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if not (hide_zeros and cm[i, j] == 0):
            ax.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=text_fontsize,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label', fontsize=text_fontsize)
    ax.set_xlabel('Predicted label', fontsize=text_fontsize)
    ax.grid('off')

    return ax
