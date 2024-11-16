"""
The :mod:`~scikitplot.kds` module includes plots for machine learning
evaluation decile analysis e.g. Gain, Lift and Decile charts, etc.

References
----------
[1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py

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
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scikitplot.utils.validation import (
  validate_plotting_kwargs_decorator,
  validate_shapes_decorator,
  validate_y_true_decorator,
  validate_y_probas_decorator,
  validate_y_probas_bounds_decorator,
)
from scikitplot.utils._helpers import (
  validate_labels,
  cumulative_gain_curve,
  binary_ks_curve,
)

from scikitplot import _api, _docstring, _preprocess


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
  'print_labels',
  'decile_table',
  'plot_cumulative_gain',
  'plot_lift',
  'plot_lift_decile_wise',
  'plot_ks_statistic',
  'report',
]


def print_labels(**kwargs):
    """
    A legend for the abbreviations of decile table column names.

    .. versionadded:: 0.3.9
        
    See Also
    --------
    decile_table : Generates the Decile Table from labels and probabilities.
    
    References
    ----------
    [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L5

    Examples
    --------
    
    .. jupyter-execute::

        >>> import scikitplot as skplt
        >>> skplt.kds.print_labels()
    """
    print(
        "LABELS INFO:\n\n",
        "prob_min         : Minimum probability in a particular decile\n", 
        "prob_max         : Minimum probability in a particular decile\n",
        "prob_avg         : Average probability in a particular decile\n",
        "cnt_events       : Count of events in a particular decile\n",
        "cnt_resp         : Count of responders in a particular decile\n",
        "cnt_non_resp     : Count of non-responders in a particular decile\n",
        "cnt_resp_rndm    : Count of responders if events assigned randomly in a particular decile\n",
        "cnt_resp_wiz     : Count of best possible responders in a particular decile\n",
        "resp_rate        : Response Rate in a particular decile [(cnt_resp/cnt_cust)*100]\n",
        "cum_events       : Cumulative sum of events decile-wise \n",
        "cum_resp         : Cumulative sum of responders decile-wise \n",
        "cum_resp_wiz     : Cumulative sum of best possible responders decile-wise \n",
        "cum_non_resp     : Cumulative sum of non-responders decile-wise \n",
        "cum_events_pct   : Cumulative sum of percentages of events decile-wise \n",
        "cum_resp_pct     : Cumulative sum of percentages of responders decile-wise \n",
        "cum_resp_pct_wiz : Cumulative sum of percentages of best possible responders decile-wise \n",
        "cum_non_resp_pct : Cumulative sum of percentages of non-responders decile-wise \n",
        "KS               : KS Statistic decile-wise \n",
        "lift             : Cumuative Lift Value decile-wise",
    )


@validate_shapes_decorator
@validate_y_true_decorator
@validate_y_probas_decorator
@validate_y_probas_bounds_decorator
def decile_table(
  ## default params
  y_true,
  y_probas,
  *args,
  pos_label=None,      # for y_true
  class_index=1,       # for y_probas
  change_deciles=10,
  digits=3,
  labels=True,
  ## additional params
  **kwargs,
):
    """
    Generates the Decile Table from labels and probabilities
    
    The Decile Table is creared by first sorting the customers by their predicted 
    probabilities, in decreasing order from highest (closest to one) to 
    lowest (closest to zero). Splitting the customers into equally sized segments, 
    we create groups containing the same numbers of customers, for example, 10 decile 
    groups each containing 10% of the customer base.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct/actual) target values.

    y_probas : array-like, shape (n_samples, n_classes)
        Prediction probabilities for each class returned by a classifier/algorithm.

    pos_label : scalar, optional
        The positive label for binary classification. If None, it defaults to 
        `classes[1]`.

        .. versionadded:: 0.3.9
        
    class_index : int, optional
        Index of the class for which to extract probabilities in multi-class case.
        If None, returns all class probabilities in the 2D case. Ignored if y_probas is 1D.

        .. versionadded:: 0.3.9

    change_deciles : int, optional, default=10
        The number of partitions for creating the table. Defaults to 10 for deciles.

    digits : int, optional, default=6
        The decimal precision for the result.

        .. versionadded:: 0.3.9

    labels : bool, optional, default=True
        If True, prints a legend for the abbreviations of decile table column names.
        
    **kwargs : dict, optional
         
        .. versionadded:: 0.3.9

    Returns
    -------
    pandas.DataFrame
        The dataframe `dt` (decile-table) with the deciles and related information.
        
    See Also
    --------
    print_labels : A legend for the abbreviations of decile table column names.

    References
    ----------
    [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L32

    Examples
    --------
    
    .. jupyter-execute::

        >>> from sklearn.datasets import load_breast_cancer as data_2_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> import scikitplot as skplt
        >>> X, y = data_2_classes(return_X_y=True, as_frame=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> skplt.kds.decile_table(
        >>>     y_test, y_prob, class_index=1
        >>> )
    """
    # Convert input to numpy arrays for efficient processing

    # Create DataFrame
    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_prob'] = y_probas
  
    # df['decile']=pd.qcut(df['y_prob'], 10, labels=list(np.arange(10,0,-1))) 
    # ValueError: Bin edges must be unique
    # Sort by probabilities
    df = df.sort_values(by=['y_prob'], ascending=False)
    df['decile'] = np.linspace(1, change_deciles+1, len(df), False, dtype=int)

    # Apply the groupby operation
    # dt abbreviation for decile_table
    dt = df.groupby('decile', group_keys=False).apply(
        lambda x: pd.Series(
            [
                np.min(x['y_prob']),
                np.max(x['y_prob']),
                np.mean(x['y_prob']),
                np.size(x['y_prob']),
                np.sum(x['y_true']),
                np.size(x['y_true'][x['y_true'] == 0]),
            ],
            index=(
                ["prob_min", "prob_max", "prob_avg", "cnt_cust", "cnt_resp", "cnt_non_resp"]
            ),
        ),
        include_groups=False,  # Deprecated since version 2.2.0: Setting include_groups to True is deprecated.
    ).reset_index()

    # Round the results
    dt['prob_min']=dt['prob_min'].round(digits)
    dt['prob_max']=dt['prob_max'].round(digits)
    dt['prob_avg']=round(dt['prob_avg'],digits)
    # dt=dt.sort_values(by='decile',ascending=False).reset_index(drop=True)

    # Calculate additional columns
    tmp = df[['y_true']].sort_values('y_true', ascending=False)
    tmp['decile'] = np.linspace(1, change_deciles+1, len(tmp), False, dtype=int)

    dt['cnt_resp_rndm'] = np.sum(df['y_true']) / change_deciles
    dt['cnt_resp_wiz'] = tmp.groupby(
      'decile',
      group_keys=False,  # Changed in version 2.0.0: group_keys now defaults to True.
      as_index=True,
    )['y_true'].sum()#['y_true']

    dt['resp_rate'] = round(dt['cnt_resp'] * 100 / dt['cnt_cust'], digits)
    dt['cum_cust'] = np.cumsum(dt['cnt_cust'])
    dt['cum_resp'] = np.cumsum(dt['cnt_resp'])
    dt['cum_resp_wiz'] = np.cumsum(dt['cnt_resp_wiz'])
    dt['cum_non_resp'] = np.cumsum(dt['cnt_non_resp'])
    dt['cum_cust_pct'] = round(dt['cum_cust'] * 100 / np.sum(dt['cnt_cust']), digits)
    dt['cum_resp_pct'] = round(dt['cum_resp'] * 100 / np.sum(dt['cnt_resp']), digits)
    dt['cum_resp_pct_wiz'] = round(dt['cum_resp_wiz'] * 100 / np.sum(dt['cnt_resp_wiz']), digits)
    dt['cum_non_resp_pct'] = round(
        dt['cum_non_resp'] * 100 / np.sum(dt['cnt_non_resp']), digits)
    dt['KS'] = round(dt['cum_resp_pct'] - dt['cum_non_resp_pct'], digits)
    dt['lift'] = round(dt['cum_resp_pct'] / dt['cum_cust_pct'], digits)

    if labels is True:
        print_labels()

    return dt


@_preprocess._preprocess_data(
  replace_names=["y_true", "y_probas",],
  # label_namer="y",  # for label params
)
@validate_plotting_kwargs_decorator
@_docstring.interpd
def plot_lift(
  ## default params
  y_true,
  y_probas,
  *args,
  pos_label=None,      # for y_true
  class_index=1,       # for y_probas
  # class_names=None,
  # multi_class=None,
  # to_plot_class_index=None,
  ## plotting params
  title='Lift Curves',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium",
  # cmap=None,
  # show_labels=True,
  # plot_micro=False,
  # plot_macro=False,
  ## additional params
  **kwargs,
):
    """\
    Generates the Decile based cumulative Lift Plot from labels and probabilities.
    
    .. dropdown:: View aliases

        **Main aliases**
        
        `scikitplot.api.kds.plot_lift`

        **Compat aliases**
        
        `scikitplot.kds.plot_lift`

    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
    The implementation here works only for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the `class_index`.

    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification. Ignored for
        binary classification.

    title : str, default='Lift Curves'
        Title of the plot.

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.
        
    **kwargs : dict, optional
         
        .. versionadded:: 0.3.9

    Other Parameters
    ----------------    
    %(_validate_plotting_kwargs_doc)s

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted lift curves.
        
    See Also
    --------
    plot_lift_decile_wise : Generates the Decile-wise Lift Plot from labels and probabilities.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Lift Curves
    
        >>> from sklearn.datasets import load_iris as data_3_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> import scikitplot as skplt
        >>> X, y = data_3_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = LogisticRegression(max_iter=int(1e5), random_state=0).fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.kds.plot_lift(
        >>>     y_val, y_probas, class_index=1,
        >>> );
    """
    # Convert input to numpy arrays for efficient processing

    pl = decile_table(y_true, y_probas, labels=False)

    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    
    # Proceed with your plotting logic here, e.g.:
    ax.plot(pl.decile.values, pl.lift.values, marker='o', label='Model')    
    # plt.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
    ax.plot([1, 10], [1, 1], 'k--', marker='o', label='Random')
    
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Deciles', fontsize=text_fontsize)
    plt.ylabel('Lift', fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)    
    # plt.show()
    return ax


@_preprocess._preprocess_data(
  replace_names=["y_true", "y_probas",],
  # label_namer="y",  # for label params
)
@validate_plotting_kwargs_decorator
@_docstring.interpd
def plot_lift_decile_wise(
  ## default params
  y_true,
  y_probas,
  *,
  pos_label=None,      # for y_true
  class_index=1,       # for y_probas
  # class_names=None,
  # multi_class=None,
  # to_plot_class_index=None,
  ## plotting params
  title='Decile-wise Lift Plot',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium",
  # cmap=None,
  # show_labels=True,
  # plot_micro=False,
  # plot_macro=False,
  ## additional params
  **kwargs,
):
    """
    Generates the Decile-wise Lift Plot from labels and probabilities

    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
    The implementation here works only for binary classification.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like, shape (n_samples, n_classes)
        Prediction probabilities for each class returned by a classifier.

    title : str, optional, default='Decile-wise Lift Plot'
        Title of the generated plot.

    title_fontsize : str or int, optional, default=14
        Font size for the plot title. Use e.g., "small", "medium", "large" or integer-values
        (8, 10, 12, etc.).

    text_fontsize : str or int, optional, default=10
        Font size for the text in the plot. Use e.g., "small", "medium", "large" or integer-values
        (8, 10, 12, etc.).

    figsize : tuple of int, optional, default=None
        Tuple denoting figure size of the plot (e.g., (6, 6)).

        .. versionadded:: 0.3.9

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted Decile-wise Lift curves.
        
    See Also
    --------
    plot_lift : Generates the Decile based cumulative Lift Plot from labels and probabilities.

    References
    ----------
    [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L190

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Lift Decile Wise Curves
    
        >>> import scikitplot as skplt
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> clf = DecisionTreeClassifier(max_depth=1, random_state=0)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> skplt.kds.plot_lift_decile_wise(y_test, y_prob, class_index=1)
    """
    # Convert input to numpy arrays for efficient processing
        
    pldw = decile_table(y_true, y_probas, labels=False)

    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    
    # Proceed with your plotting logic here, e.g.:
    ax.plot(pldw.decile.values, pldw.cnt_resp.values / pldw.cnt_resp_rndm.values, marker='o', label='Model')
    # plt.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
    ax.plot([1, 10], [1, 1], 'k--', marker='o', label='Random')
    
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Deciles', fontsize=text_fontsize)
    plt.ylabel('Lift @ Decile', fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)    
    # plt.show()
    return ax


@_preprocess._preprocess_data(
  replace_names=["y_true", "y_probas",],
  # label_namer="y",  # for label params
)
@validate_plotting_kwargs_decorator
@_docstring.interpd
def plot_cumulative_gain(
  ## default params
  y_true,
  y_probas,
  *,
  pos_label=None,      # for y_true
  class_index=1,       # for y_probas
  # class_names=None,
  # multi_class=None,
  # to_plot_class_index=None,
  ## plotting params
  title='Cumulative Gain Plot',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium",
  # cmap=None,
  # show_labels=True,
  # plot_micro=False,
  # plot_macro=False,
  ## additional params
  **kwargs,
):
    """
    Generates the Decile-wise Lift Plot from labels and probabilities

    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
    The implementation here works only for binary classification.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    
    y_probas : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the `class_index`.
    
    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification. Ignored for
        binary classification.
    
    title : str, optional, default='Cumulative Gain Curves'
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
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted cumulative gain curves.
    
    Notes
    -----
    The implementation is specific to binary classification.
    
    References
    ----------
    [1] http://mlwiki.org/index.php/Cumulative_Gain_Chart
    
    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Cumulative Gain Curves
    
        >>> from sklearn.datasets import load_iris as data_3_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> import scikitplot as skplt
        >>> X, y = data_3_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = LogisticRegression(max_iter=int(1e5), random_state=0).fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.kds.plot_cumulative_gain(
        >>>     y_val, y_probas, class_index=1,
        >>> );
    """
    # Convert input to numpy arrays for efficient processing
        
    pcg = decile_table(y_true, y_probas, labels=False)

    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    
    # Proceed with your plotting logic here, e.g.:
    ax.plot(np.append(0, pcg.decile.values), np.append(0, pcg.cum_resp_pct.values), marker='o', label='Model')
    ax.plot(np.append(0, pcg.decile.values), np.append(0, pcg.cum_resp_pct_wiz.values), 'c--', label='Wizard')
    # plt.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
    ax.plot([0, 10], [0, 100], 'k--', marker='o', label='Random')
    
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Deciles', fontsize=text_fontsize)
    plt.ylabel('% Resonders', fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)    
    # plt.show()
    return ax


@_preprocess._preprocess_data(
  replace_names=["y_true", "y_probas",],
  # label_namer="y",  # for label params
)
@validate_plotting_kwargs_decorator
@_docstring.interpd
def plot_ks_statistic(
  ## default params
  y_true,
  y_probas,
  *,
  pos_label=None,      # for y_true
  class_index=1,       # for y_probas
  # class_names=None,
  # multi_class=None,
  # to_plot_class_index=None,
  ## plotting params
  title='KS Statistic Plot',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium",
  digits=2,
  ## additional params
  **kwargs,
):
    """
    Generates the KS Statistic Plot from labels and probabilities

    Kolmogorov-Smirnov (KS) statistic is used to measure how well the 
    binary classifier model separates the Responder class (Yes) from 
    Non-Responder class (No). The range of K-S statistic is between 0 and 1. 
    Higher the KS statistic value better the model in separating the 
    Responder class from Non-Responder class.

    Parameters
    ----------
    y_true : array-like, shape (n_samples)
        Ground truth (correct) target values.

    y_probas : array-like, shape (n_samples, n_classes)
        Prediction probabilities for each class returned by a classifier.
    
    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification. Ignored for
        binary classification.

    title : str, optional, default='KS Statistic Plot'
        Title of the generated plot.

    ax : list of matplotlib.axes.Axes, optional, default=None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).
        Axes like ``fig.add_subplot(1, 1, 1)`` or ``plt.gca()``

    fig : matplotlib.pyplot.figure, optional, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

        .. versionadded:: 0.3.9

    figsize : tuple of 2 ints, optional
        Tuple denoting figure size of the plot e.g. (6, 6). Defaults to None.

    title_fontsize : str or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or integer-values. Defaults to "large".

    text_fontsize : str or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or integer-values. Defaults to "medium".

    digits : int, optional
        Number of digits for formatting output floating point values. Use e.g. 2 or 4. Defaults to 2.

        .. versionadded:: 0.3.9

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was drawn.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: KS Statistic Plot
    
        >>> from sklearn.datasets import load_breast_cancer as data_2_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> import scikitplot as skplt
        >>> X, y = data_2_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = LogisticRegression(max_iter=int(1e5), random_state=0).fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.kds.plot_ks_statistic(
        >>>     y_val, y_probas, class_index=1,
        >>> );
    """
    # Convert input to numpy arrays for efficient processing
        
    pks = decile_table(y_true, y_probas, labels=False)

    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    
    # Proceed with your plotting logic here, e.g.:
    ax.plot(np.append(0, pks.decile.values), np.append(0, pks.cum_resp_pct.values),
             marker='o', label='Responders')
    ax.plot(np.append(0, pks.decile.values), np.append(0, pks.cum_non_resp_pct.values),
             marker='o', label='Non-Responders')
    # ax.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
    ksmx = pks.KS.max()
    ksdcl = pks[pks.KS == ksmx].decile.values
    ax.plot([ksdcl, ksdcl],
             [pks[pks.KS == ksmx].cum_resp_pct.values,
              pks[pks.KS == ksmx].cum_non_resp_pct.values],
             'g--', marker='o', label='KS Statisic: ' + str(ksmx) + ' at decile ' + str(list(ksdcl)[0]))
    
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Deciles', fontsize=text_fontsize)
    plt.ylabel('% Resonders', fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)    
    # plt.show()
    return ax


@_preprocess._preprocess_data(
  replace_names=["y_true", "y_probas",],
  # label_namer="y",  # for label params
)
@_docstring.interpd
def report(
  ## default params
  y_true,
  y_probas,
  *,
  pos_label=None,      # for y_true
  class_index=1,       # for y_probas
  # class_names=None,
  # multi_class=None,
  display_term_tables=True,
  digits=3,
  ## plotting params
  ax=None,
  fig=None,
  figsize=(12, 7),
  title_fontsize="large",
  text_fontsize="medium",
  plot_style = None,
  ## additional params
  **kwargs,
):
    """    
    Generates a decile table and four plots:
    
    - ``Lift`` -> :func:`~scikitplot.kds.plot_lift`
    - ``Lift@Decile`` -> :func:`~scikitplot.kds.plot_lift_decile_wise`
    - ``Gain`` -> :func:`~scikitplot.kds.plot_cumulative_gain`
    - ``KS`` -> :func:`~scikitplot.kds.plot_ks_statistic`
    
    from labels and probabilities.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like, shape (n_samples, n_classes)
        Prediction probabilities for each class returned by a classifier.
    
    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification. Ignored for
        binary classification.

    labels : bool, optional, default=True
        If True, prints a legend for the abbreviations of decile table column names.

        .. deprecated:: 0.3.9
            This parameter is deprecated and will be removed in version 0.5.0. Use
            ``display_term_tables`` instead.

    display_term_tables : bool, optional, default=True
        If True, prints a legend for the abbreviations of decile table column names.

        .. versionadded:: 0.3.9

    ax : matplotlib.axes.Axes, optional, default=None
        The axes upon which to plot. If None, a new set of axes is created.

    figsize : tuple of int, optional, default=None
        Tuple denoting figure size of the plot (e.g., (6, 6)).

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title. Use e.g., "small", "medium", "large" or integer-values.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot. Use e.g., "small", "medium", "large" or integer-values.

    digits : int, optional, default=3
        Number of digits for formatting output floating point values. Use e.g., 2 or 4.

        .. versionadded:: 0.3.9

    plot_style : str, optional, default=None
        Check available styles with "plt.style.available". Examples include:
        ['ggplot', 'seaborn', 'bmh', 'classic', 'dark_background', 'fivethirtyeight', 
        'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 
        'seaborn-dark-palette', 'tableau-colorblind10', 'fast'].

        .. versionadded:: 0.3.9

    Returns
    -------
    pandas.DataFrame
        The dataframe containing the decile table with the deciles and related information.
        
    See Also
    --------
    plot_lift : Generates the Decile based cumulative Lift Plot from labels and probabilities.

    plot_lift_decile_wise : Generates the Decile-wise Lift Plot from labels and probabilities.

    plot_cumulative_gain : Generates the cumulative Gain Plot from labels and probabilities.

    plot_ks_statistic : Generates the Kolmogorov-Smirnov (KS) Statistic Plot from labels and probabilities.

    print_labels : A legend for the abbreviations of decile table column names.

    decile_table : Generates the Decile Table from labels and probabilities.

    References
    ----------
    [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L382

    Examples
    --------
        
    .. jupyter-execute::

        >>> from sklearn.datasets import load_breast_cancer as data_2_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> import scikitplot as skplt
        >>> X, y = data_2_classes(return_X_y=True, as_frame=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> dt = skplt.kds.report(
        >>>     y_test, y_prob, class_index=1
        >>> )
        >>> dt
    """
    # Convert input to numpy arrays for efficient processing    
    dc = decile_table(
        y_true,
        y_probas,
        labels=display_term_tables,
        round_decimal=digits
    )

    ##################################################################
    ## Plotting
    ##################################################################
    if plot_style is None:
        None
    else:
        plt.style.use(plot_style)
    
    fig = plt.figure(figsize=figsize)

    # Cumulative Lift Plot
    ax = plt.subplot(2, 2, 1)
    plot_lift(y_true, y_probas, fig=fig, ax=ax)

    #  Decile-wise Lift Plot
    ax = plt.subplot(2, 2, 2)
    plot_lift_decile_wise(y_true, y_probas, fig=fig, ax=ax)

    # Cumulative Gains Plot
    ax = plt.subplot(2, 2, 3)
    plot_cumulative_gain(y_true, y_probas, fig=fig, ax=ax)

    # KS Statistic Plot
    ax = plt.subplot(2, 2, 4)
    plot_ks_statistic(y_true, y_probas, fig=fig, ax=ax)

    fig.tight_layout()
    return (dc)