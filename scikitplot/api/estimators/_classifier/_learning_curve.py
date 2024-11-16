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

from sklearn.model_selection import learning_curve

from scikitplot.utils.validation import (
  validate_plotting_kwargs_decorator,
  # validate_shapes_decorator,
  # validate_y_true_decorator,
  # validate_y_probas_decorator,
  # validate_y_probas_bounds_decorator,
)


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
  'plot_learning_curve',
]


@validate_plotting_kwargs_decorator
def plot_learning_curve(
  ## default params
  estimator,
  X,
  y,
  *,
  # groups=None,
  train_sizes=None,
  cv=None,
  scoring=None,
  # exploit_incremental_learning=False,
  n_jobs=None,
  # pre_dispatch="all",
  verbose=0,
  shuffle=False,
  random_state=None,
  # error_score=np.nan,
  # return_times=False,
  fit_params=None,
  ## plotting params
  title='Learning Curves',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium",
  ## additional params
  **kwargs,
):
    """
    Generates a plot of the train and test learning curves for a classifier.

    The learning curves plot the performance of a classifier as a function of the number of
    training samples. This helps in understanding how well the classifier performs
    with different amounts of training data.

    Parameters
    ----------
    estimator : object type that implements the "fit" method
        An object of that type which is cloned for each validation. It must
        also implement "predict" unless `scoring` is a callable that doesn't
        rely on "predict" to compute a score.

    X : array-like, shape (n_samples, n_features)
        Training data, where `n_samples` is the number of samples
        and `n_features` is the number of features.

    y : array-like, shape (n_samples,) or (n_samples, n_features), optional
        Target relative to `X` for classification or regression.
        None for unsupervised learning.

    train_sizes : iterable, optional
        Determines the training sizes used to plot the learning curve.
        If None, `np.linspace(.1, 1.0, 5)` is used.
        
    cv : int, cross-validation generator, iterable or None, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : str, callable, or None, optional, default=None
        A string (see scikit-learn model evaluation documentation)
        or a scorer callable object/function
        with signature `scorer(estimator, X, y)`.

    n_jobs : int, optional, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the different training and test sets.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    shuffle : bool, optional, default=True
        Whether to shuffle the training data before splitting using cross-validation.

    random_state : int or RandomState, optional
        Pseudo-random number generator state used for random sampling.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

        .. versionadded:: 0.3.9

    title : str, optional, default="Learning Curves"
        Title of the generated plot.

    ax : matplotlib.axes.Axes, optional, default=None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib.pyplot.figure, optional, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    figsize : tuple of int, optional, default=None
        Tuple denoting figure size of the plot, e.g., (6, 6).

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.
        Use e.g., "small", "medium", "large" or integer values.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.
        Use e.g., "small", "medium", "large" or integer values.

    kwargs: dict
        generic keyword arguments.


    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was drawn.


    .. dropdown:: References
    
      * `"scikit-learn learning_curve"
        <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#>`_.


    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Learning Curves

        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.naive_bayes import GaussianNB
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = GaussianNB()
        >>> model.fit(X_train, y_train)
        >>> y_val_pred = model.predict(X_val)
        >>> skplt.estimators.plot_learning_curve(
        >>>     model, X_val, y_val_pred,
        >>> );
    """
    #################################################
    ## Preprocessing
    #################################################    
    # Proceed with your preprocess logic here
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    train_sizes, train_scores, test_scores = learning_curve(
        ## default params
        estimator,
        X,
        y,
        # groups=None,
        train_sizes=train_sizes, 
        cv=cv,
        scoring=scoring,
        # exploit_incremental_learning=False,
        n_jobs=n_jobs,
        # pre_dispatch="all",
        verbose=verbose,
        shuffle=shuffle,
        random_state=random_state,
        # error_score=np.nan,
        # return_times=False,
        fit_params=fit_params,
    )    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Get model name if available
    model_name = f'{estimator.__class__.__name__} ' if hasattr(estimator, '__class__') else ''
    
    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    # Proceed with your plotting logic here
    ax.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color="r"
    )
    ax.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="g"
    )
    ax.plot(
        train_sizes, train_scores_mean, 'o-', color="r",
        label="Training score"
    )
    ax.plot(
        train_sizes, test_scores_mean, 'o-', color="g",
        label="Cross-validation score"
    )
    ax.set_title(model_name + title, fontsize=title_fontsize)
    ax.set_xlabel("Training examples", fontsize=text_fontsize)
    ax.set_ylabel("Score", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid()
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=text_fontsize)
    return ax