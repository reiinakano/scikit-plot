"""
The :mod:`scikitplot.estimators` module includes plots built specifically for
scikit-learn estimator (classifier/regressor) instances e.g. Random Forest.
You can use your own estimators, but these plots assume specific properties
shared by scikit-learn estimators. The specific requirements are documented per
function.
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import learning_curve


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


def plot_learning_curve(clf, X, y, title='Learning Curve', cv=None,
                        shuffle=False, random_state=None,
                        train_sizes=None, n_jobs=1, scoring=None,
                        ax=None, figsize=None, title_fontsize="large",
                        text_fontsize="medium"):
    """Generates a plot of the train and test learning curves for a classifier.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict``
            methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification or regression;
            None for unsupervised learning.

        title (string, optional): Title of the generated plot. Defaults to
            "Learning Curve"

        cv (int, cross-validation generator, iterable, optional): Determines
            the cross-validation strategy to be used for splitting.

            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is
            used.

        shuffle (bool, optional): Used when do_cv is set to True. Determines
            whether to shuffle the training data before splitting using
            cross-validation. Default set to True.

        random_state (int :class:`RandomState`): Pseudo-random number generator
            state used for random sampling.

        train_sizes (iterable, optional): Determines the training sizes used to
            plot the learning curve. If None, ``np.linspace(.1, 1.0, 5)`` is
            used.

        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to
            1.

        scoring (string, callable or None, optional): default: None
            A string (see scikit-learn model evaluation documentation) or a
            scorerbcallable object / function with signature
            scorer(estimator, X, y).

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
        >>> skplt.estimators.plot_learning_curve(rf, X, y)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_learning_curve.png
           :align: center
           :alt: Learning Curve
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Training examples", fontsize=text_fontsize)
    ax.set_ylabel("Score", fontsize=text_fontsize)
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring, shuffle=shuffle, random_state=random_state)
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
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="best", fontsize=text_fontsize)

    return ax
