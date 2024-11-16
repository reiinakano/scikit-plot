"""
The :py:mod:`~scikitplot.estimators` module includes plots built specifically for
scikit-learn estimator (classifier/regressor) instances e.g. Random Forest.
You can use your own estimators, but these plots assume specific properties
shared by scikit-learn estimators. The specific requirements are documented per
function.

The Scikit-plots Functional API

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
  absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
  division,         # Changes the division operator `/` to always perform true division.
  print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
  unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from scikitplot.utils.validation import (
  validate_plotting_kwargs_decorator,
  # validate_shapes_decorator,
  # validate_y_true_decorator,
  # validate_y_probas_decorator,
  # validate_y_probas_bounds_decorator,
)


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
  'plot_feature_importances',
]


@validate_plotting_kwargs_decorator
def plot_feature_importances(
  ## default params
  estimator,
  *,
  feature_names=None,
  class_index=None,
  threshold=None,
  ## plotting params
  title='Feature Importances',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium",
  cmap='PiYG',
  # bar plot
  order=None,
  orientation='vertical',
  x_tick_rotation=None,
  bar_padding=11,
  display_bar_label=True,
  digits=4,
  ## additional params
  **kwargs,
):
    """
    Generates a plot of a sklearn model's feature importances.

    This function handles different types of classifiers and their respective
    feature importances (``feature_importances_``) or coefficient (``coef_``) attributes,
    if not provide its compute sklearn permutation importances.
    It supports models wrapped in pipelines.

    Supports models like:
    
    - :py:class:`~sklearn.linear_model.LinearRegression`
    - :py:class:`~sklearn.linear_model.LogisticRegression`
    - :py:class:`~sklearn.neighbors.KNeighborsClassifier`
    - :py:class:`~sklearn.svm.LinearSVC`
    - :py:class:`~sklearn.svm.SVC`
    - :py:class:`~sklearn.tree.DecisionTreeClassifier`
    - :py:class:`~sklearn.ensemble.RandomForestClassifier`
    - :py:class:`~sklearn.decomposition.PCA`
    - `xgboost Python API <https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn>`_
    - `catboost Python API <https://catboost.ai/en/docs/concepts/python-quickstart>`_

    Parameters
    ----------
    estimator : estimator object        
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a classifier.

    feature_names : list of str, optional, default=None
        List of feature names corresponding to the features. If None, feature
        indices are used.

    class_index : int, optional, default=None
        Index of the class of interest for multi-class classification.
        Defaults to None.

    threshold : float, optional, default=None
        Threshold for filtering features by absolute importance. Only
        features with an absolute importance greater than this threshold will
        be plotted. Defaults to None (plot all features).

    title : str, optional, default='Feature Importances'
        Title of the generated plot.

    ax : matplotlib.axes.Axes, optional, default=None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib.pyplot.figure, optional, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    figsize : tuple, optional, default=None
        Tuple denoting figure size of the plot e.g. (6, 6)

    title_fontsize : str or int, optional, default='large'
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or
        integer-values.

    text_fontsize : str or int, optional, default='medium'
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or
        integer-values.

    cmap : None, str or matplotlib.colors.Colormap, optional, default='PiYG'
        Colormap used for plotting.
        Options include 'viridis', 'PiYG', 'plasma', 'inferno', etc.
        See Matplotlib Colormap documentation for available choices.
        - https://matplotlib.org/stable/users/explain/colors/index.html

    order : {'ascending', 'descending', None}, optional, default=None
        Order of feature importance in the plot. Defaults to None
        (automatically set based on orientation).

    orientation : {'vertical' | 'v' | 'y', 'horizontal' | 'h' | 'y'}, optional
        Orientation of the bar plot. Defaults to 'vertical'.

    x_tick_rotation : int, optional, default=None
        Rotates x-axis tick labels by the specified angle. Defaults to None
        (automatically set based on orientation).

    bar_padding : float, optional, default=11
        Padding between bars in the plot.

    display_bar_label : bool, optional, default=True
        Whether to display the bar labels.

    digits : int, optional, default=4
        Number of digits for formatting AUC values in the plot.

        .. versionadded:: 0.3.9

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Feature Importances
    
       >>> from sklearn.datasets import load_digits as data_10_classes
       >>> from sklearn.model_selection import train_test_split
       >>> from sklearn.ensemble import RandomForestClassifier
       >>> import scikitplot as skplt
       >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
       >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
       >>> model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
       >>> skplt.estimators.plot_feature_importances(
       >>>     model,
       >>>     orientation='y',
       >>>     figsize=(11, 5),
       >>> );
    """    
    #################################################
    ## Preprocessing
    #################################################
    # Proceed with your preprocess logic here
    
    # Handle pipelines
    if hasattr(estimator, 'named_steps'):
        estimator = estimator.named_steps[next(reversed(estimator.named_steps))]

    # Determine the appropriate attribute for feature importances or coefficients
    if hasattr(estimator, 'feature_importances_'):
        importances = np.asarray(estimator.feature_importances_)
    # LDA (scikit-learn < 0.24)
    elif hasattr(estimator, 'coef_'):
        if estimator.coef_.ndim > 1:  # Multi-class case
            if class_index is None:
                importances = np.mean(np.abs(estimator.coef_), axis=0)
            else:
                importances = np.asarray(estimator.coef_)[class_index]
        else:
            importances = np.asarray(estimator.coef_).ravel()
    # PCA
    elif hasattr(estimator, 'explained_variance_ratio_'):
        importances = np.asarray(estimator.explained_variance_ratio_)
    else:
        raise TypeError(
            'The estimator does not have an attribute for feature '
            'importances or coefficients.'
        )
    # Obtain feature names
    if feature_names is None:
        # sklearn models
        if hasattr(estimator, 'feature_names_in_'):
            feature_names = np.asarray(estimator.feature_names_in_, dtype=object)
        # catboost
        elif hasattr(estimator, 'feature_names_'):
            feature_names = np.asarray(estimator.feature_names_, dtype=object)
        else:
            # Ensure feature_names are strings
            feature_names = np.asarray(np.arange(len(importances), dtype=int), dtype=object)
    else:
        feature_names = np.asarray(feature_names, dtype=object)

    # Generate indices
    indices = np.arange(len(importances))

    # Apply filtering based on the threshold
    if threshold is not None:
        mask = np.abs(importances) > threshold
        indices = indices[mask].copy()

    # Apply ordering based on orientation
    if order is None:
        order = 'ascending' if orientation in ['horizontal', 'h', 'x'] else 'descending'
    
    if order == 'descending':
        # Sort the indices based on the importances in descending order
        sorted_indices = np.argsort(importances[indices])[::-1]
        indices = indices[sorted_indices].copy()
    elif order == 'ascending':
        # Sort the indices based on the importances in ascending order
        sorted_indices = np.argsort(importances[indices])
        indices = indices[sorted_indices].copy()

    # Reorder the importances array according to the sorted indices
    importances = importances[indices].copy()
    # Reorder the feature names according to the sorted indices
    feature_names = feature_names[indices].copy()

    # Get model name if available
    model_name = f'{estimator.__class__.__name__} ' if hasattr(estimator, '__class__') else ''
    
    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    # Proceed with your plotting logic here
    # Plot bars based on orientation
    for idx, (col, imp) in enumerate(zip(feature_names, importances)):
        # Default colormap if not provided, 'viridis'
        color = plt.get_cmap(cmap)( float(idx) / len(importances) )
        if orientation in ['vertical', 'v', 'y']:
            bar = ax.bar(x=str(col), height=imp, color=color)
        elif orientation in ['horizontal', 'h', 'x']:
            bar = ax.barh(y=str(col), width=imp, color=color)
        else:
            raise ValueError(
                "Invalid value for orientation: "
                "must be ['vertical', 'v', 'y'] or ['horizontal', 'h', 'x']."
            )

    # Set default x_tick_rotation based on orientation
    if x_tick_rotation is None:
        x_tick_rotation = (
            0 if orientation in ['horizontal', 'h', 'x'] else 90
        )

    if display_bar_label:
        for bars in ax.containers:
            ax.bar_label(
                bars,
                fmt=lambda x:'{:0>{digits}.{digits}f}'.format(x, digits=digits),
                fontsize=text_fontsize, 
                rotation=x_tick_rotation,
                padding=bar_padding,
            )
    ax.set_title(model_name + title, fontsize=title_fontsize)
    if orientation in ['vertical', 'v', 'y']:
        # Set x-ticks positions and labels
        ax.set_xticks(np.arange(len(feature_names)))
        # ax.set_xticklabels(feature_names, rotation=x_tick_rotation, fontsize=text_fontsize)
        ax.tick_params(axis='x', rotation=x_tick_rotation, labelsize=text_fontsize)
        ax.set_xlabel("Features | Index", fontsize=text_fontsize)
        ax.set_ylabel("Importance", fontsize=text_fontsize)
    elif orientation in ['horizontal', 'h', 'x']:
        # Set y-ticks positions and labels
        ax.set_yticks(np.arange(len(feature_names)))
        # ax.set_yticklabels(feature_names, rotation=x_tick_rotation, fontsize=text_fontsize)
        ax.tick_params(axis='y', rotation=x_tick_rotation, labelsize=text_fontsize)
        ax.set_xlabel("Importance", fontsize=text_fontsize)
        ax.set_ylabel("Features | Index", fontsize=text_fontsize)
    else:
        raise ValueError(
            "Invalid value for orientation: must be "
            "must be ['vertical', 'v', 'y'] or ['horizontal', 'h', 'x']."
        )
    
    # Adjust plot limits if needed
    if orientation in ['vertical', 'v', 'y']:
        # Increase the upper limit by 15%
        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] * 1.2])
    elif orientation in ['horizontal', 'h', 'x']:
        # Increase the upper limit by 15%
        ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] * 1.2])
    
    plt.legend([f'n_features_out_: {len(importances)}'])
    plt.tight_layout()
    return ax, feature_names