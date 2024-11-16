"""
The :mod:`~scikitplot.metrics` module includes plots for machine learning
evaluation metrics e.g. confusion matrix, silhouette scores, etc.

The Scikit-plots Functional API

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
  absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
  division,         # Changes the division operator `/` to always perform true division.
  print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
  unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import warnings
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
  label_binarize, LabelEncoder,
  MinMaxScaler,
)
from sklearn import metrics
from sklearn.metrics import (
  confusion_matrix,
  classification_report,
  auc,
  roc_curve,
  precision_recall_curve,
  average_precision_score,
  silhouette_score,
  silhouette_samples,
)
from sklearn.utils.multiclass import unique_labels
from sklearn.calibration import calibration_curve
from sklearn.utils import deprecated

# Sigmoid and Softmax functions
import scipy.special as sc
from scipy.special import (
  expit as sigmoid,
  softmax  
)

from scikitplot.utils.validation import (
  validate_plotting_kwargs_decorator,
  validate_shapes_decorator,
  validate_y_true_decorator,
  validate_y_probas_decorator,
  validate_y_probas_bounds_decorator,
)


## Define __all__ to specify the public interface of the module,
# not required default all above func
__all__ = [
  'plot_precision_recall_curve',
  'plot_precision_recall',
]


@deprecated(
    'This will be removed in v0.5.0. Please use '
    'scikitplot.metrics.plot_precision_recall instead.'
)
def plot_precision_recall_curve(
    y_true, y_probas,
    title='Precision-Recall Curve',
    curves=('micro', 'each_class'), ax=None,
    figsize=None, cmap='nipy_spectral',
    title_fontsize="large",
    text_fontsize="medium"
):
    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Precision-Recall curve".

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "each_class")`
            i.e. "micro" for micro-averaged curve

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
        >>> nb = GaussianNB()
        >>> nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_precision_recall_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: /images/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if 'micro' not in curves and 'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro" or "each_class"')

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true, probas[:, i], pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i],
                                                       probas[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
        y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                           average='micro')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall[i], precision[i], lw=2,
                    label='Precision-recall curve of class {0} '
                          '(area = {1:0.3f})'.format(classes[i],
                                                     average_precision[i]),
                    color=color)

    if 'micro' in curves:
        ax.plot(recall[micro_key], precision[micro_key],
                label='micro-average Precision-recall curve '
                      '(area = {0:0.3f})'.format(average_precision[micro_key]),
                color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=text_fontsize)
    return ax


@validate_plotting_kwargs_decorator
@validate_shapes_decorator
@validate_y_true_decorator
@validate_y_probas_decorator
@validate_y_probas_bounds_decorator
def plot_precision_recall(
  ## default params
  y_true,
  y_probas,
  *,
  pos_label=None,      # for binary y_true
  class_index=None,    # for multi-class y_probas
  class_names=None,
  multi_class=None,
  to_plot_class_index=None,
  ## plotting params
  title='Precision-Recall AUC Curves',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium",
  cmap=None,
  show_labels=True,
  digits=4,
  plot_micro=True,
  plot_macro=False,
  pr_auc='pr_auc',
  ap_score=True,
  plot_chance_level=True,
  ## additional params
  **kwargs,
):
    """
    Generates the Precision-Recall AUC Curves from labels and predicted scores/probabilities.
    
    The Precision-Recall curve plots the precision against the recall for different threshold values. 
    The area under the curve (AUC) represents the classifier's performance. This function supports 
    both binary and multiclass classification tasks.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like, shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the ``class_index``.

    class_names : list of str, optional, default=None
        List of class names for the legend. Order should match the order of classes in `y_probas`.

    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot for the entire probability distribution.

    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification.
        Ignored for binary classification.

    to_plot_class_index : list-like, optional, default=None
        Specific classes to plot. If a given class does not exist, it will be ignored. 
        If None, all classes are plotted.

    title : str, optional, default='Precision-Recall AUC Curves'
        Title of the generated plot.

    ax : list of matplotlib.axes.Axes, optional, default=None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).
        Axes like ``fig.add_subplot(1, 1, 1)`` or ``plt.gca()``

    fig : matplotlib.pyplot.figure, optional, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

        .. versionadded:: 0.3.9

    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.

    cmap : None, str or matplotlib.colors.Colormap, optional, default=None
        Colormap used for plotting.
        Options include 'viridis', 'PiYG', 'plasma', 'inferno', 'nipy_spectral', etc.
        See Matplotlib Colormap documentation for available choices.
        
        - https://matplotlib.org/stable/users/explain/colors/index.html
        - plt.colormaps()
        - plt.get_cmap()  # None == 'viridis'

    show_labels : bool, optional, default=True
        Whether to display the legend labels.

    digits : int, optional, default=3
        Number of digits for formatting PR AUC values in the plot.

        .. versionadded:: 0.3.9

    plot_micro : bool, optional, default=False
        Whether to plot the micro-average ROC AUC curve.

    plot_macro : bool, optional, default=False
        Whether to plot the macro-average ROC AUC curve.

    pr_auc : {'average_precision', 'pr_auc'}, optional, default='pr_auc'
        Area under PR AUC curve or Average precision score.
        sklearn uses default 'average_precision' both are slightly different.

        .. versionadded:: 0.3.9

    ap_score : bool, optional, default: True
        Annotate the graph with the average precision score, a summary of the
        plot that is computed as the weighted mean of precisions at each
        threshold, with the increase in recall from the previous threshold used
        as the weight.

        .. versionadded:: 0.3.9
        
    plot_chance_level : bool, optional, default: True
        Whether to plot the chance level. The chance level is the prevalence
        of the positive label. It is used for plotting the chance level line.

        .. versionadded:: 0.3.9

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted PR AUC curves.

    Notes
    -----
    The implementation is specific to binary classification. For multiclass problems, 
    the 'ovr' or 'multinomial' strategies can be used. When ``multi_class='ovr'``, 
    the plot focuses on the specified class (``class_index``).


    .. dropdown:: References
    
      * `"scikit-learn plot_precision_recall"
        <https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#>`_.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Precision-Recall AUC Curves

        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.naive_bayes import GaussianNB
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = GaussianNB()
        >>> model.fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.metrics.plot_precision_recall(
        >>>     y_val, y_probas,
        >>> );
    """
    def pr_auc_score(recall, precision, y_true, y_proba, pr_auc='pr_auc'):
        if pr_auc == 'pr_auc' :
            score = auc(
                recall, precision,
            )
        elif pr_auc == 'average_precision':
            score = average_precision_score(
                y_true, y_proba,
                #pos_label=classes[i]
            )
        else:
            raise ValueError(
                "Unsupported `pr_auc` scoring option, "
                "only 'pr_auc' or 'average_precision' is supported."
            )
        return score
    ##################################################################
    ## Preprocessing
    ##################################################################
    # Proceed with your preprocess logic here  
    # Convert input to numpy arrays for efficient processing
    # equalize ndim for y_true and y_probas 2D
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_true = np.column_stack(
            [1 - y_true, y_true]
        )
    if y_probas.ndim == 1:
        y_probas = y_probas[:, None]
        y_probas = np.column_stack(
            [1 - y_probas, y_probas]
        )          
    if (y_true.ndim == y_probas.ndim) and (y_true.shape == y_probas.shape):
        pass
    else:
        raise ValueError(
            f'Shape mismatch `y_true` shape {y_true.shape}, '
            f'`y_probas` shape {y_probas.shape}'
        )

    # Get unique classes and filter the ones to plot
    classes = np.arange(y_true.shape[1])    
    to_plot_class_index = classes if to_plot_class_index is None else to_plot_class_index
    indices_to_plot = np.isin(element=classes, test_elements=to_plot_class_index)
    
    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    # Proceed with your plotting logic here
    # Initialize dictionaries to store
    precision_dict, recall_dict = {}, {}
    average_precision_dict = {}
    line_kwargs = {"drawstyle": "steps-post"}

    # Loop for all classes to get different class
    for i, to_plot in enumerate(indices_to_plot):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            y_true[:, i], y_probas[:, i],
            # pos_label=classes[i]
        )
        # average_precision
        average_precision_dict[i] = pr_auc_score(
            recall=recall_dict[i],
            precision=precision_dict[i],
            y_true=y_true[:, i],
            y_proba=y_probas[:, i],
            pr_auc=pr_auc,
        )
        # to plot
        if to_plot:
            if class_names is None:
                class_names = classes            
            color = plt.get_cmap(cmap)( float(i) / len(classes) )
            # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_plot/precision_recall_curve.py#L190
            ax.plot(
                recall_dict[i], precision_dict[i],
                ls='-', lw=2, color=color,
                label=(
                    f'Class {classes[i]} '
                    f'(area = {average_precision_dict[i]:0>{digits}.{digits}f})'
                ),
                **line_kwargs,
            )

    # Whether or to plot macro or micro
    if plot_micro:
        precision, recall, _ = precision_recall_curve(
            y_true.ravel(), y_probas.ravel()
        )
        # average_precision
        average_precision = pr_auc_score(
            recall=recall,
            precision=precision,
            y_true=y_true.ravel(),
            y_proba=y_probas.ravel(),
            pr_auc=pr_auc,
        )
        # to plot
        ax.plot(
            recall, precision,
            ls=':', lw=3, color='deeppink',
            label=(
                'micro-average '
                f'(area = {average_precision:0>{digits}.{digits}f})'
            ),
            **line_kwargs,
        )

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_precision = np.unique(np.concatenate(
            [ precision_dict[i] for i in range(len(classes)) ]
        ))
        # Then interpolate all ROC curves at this points
        mean_recall = np.zeros_like(all_precision)
        for i in range(len(classes)):
            mean_recall += np.interp(
                all_precision, precision_dict[i], recall_dict[i]
            )
        # Finally average it
        mean_recall /= len(classes)
        
        # average_precision
        average_precision = average_precision_score(
            y_true.ravel(), y_probas.ravel(), average='macro'
        )
        ax.plot(
            mean_recall, all_precision,
            ls=':', lw=3, color='navy',
            label=(
                'macro-average '
                f'(area = {average_precision:0>{digits}.{digits}f})'
            ),
            **line_kwargs,
        )
        
    if ap_score:
        average_precision = average_precision_score(
            y_true.ravel(), y_probas.ravel()
        )
        label = "Avg. precision={:0>{digits}.{digits}f}".format(average_precision, digits=digits)
        ax.axhline(y=average_precision, color="r", ls="--", label=label)
        
    if plot_chance_level:
        # chance_level_line
        class_count = collections.Counter(y_true.ravel())
        prevalence_pos_label = class_count[class_index] / sum(class_count.values())
        label = f"Chance level (AP = {prevalence_pos_label:0>{2}.{2}f})"  # digits
        ax.plot(
            [0, 1], [prevalence_pos_label, prevalence_pos_label],
            ls='--', lw=1, c='k',
            label = label,
        )

    # Plot the baseline, label='Baseline'
    ax.plot([0, 1], [1, 0], ls='--', lw=1, c='gray')

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Recall', fontsize=text_fontsize)
    ax.set_ylabel('Precision', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xlim([-0.0, 1.025])
    ax.set_ylim([-0.0, 1.050])

    # Define the desired number of ticks
    # num_ticks = 10

    ## Set x-axis ticks and labels
    ## ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator( (ax.get_xlim()[1] / 10) ))
    # ax.xaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    # ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    # ax.yaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    # ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )

    # Enable grid and display legend
    ax.grid(True)
    if show_labels:        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc='lower left', 
                fontsize=text_fontsize,
                title=f'PR-AUC by {pr_auc}' + (' One-vs-Rest (OVR)' if multi_class == 'ovr' else ''),
                alignment='left'
            )
        
    # equalize the scales
    # plt.axis('equal')
    # ax.set_aspect(aspect='equal', adjustable='box')    
    plt.tight_layout()
    return ax