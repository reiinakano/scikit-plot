from __future__ import absolute_import
import warnings
import types
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


def classifier_factory(clf):
    """Takes a scikit-learn classifier instance and embeds scikit-plot instance methods in it.

    Args:
        clf: Scikit-learn classifier instance

    Returns:
        plot_clf: The scikit-learn classifier with embedded scikit-plot instance methods.

    Raises:
        ValueError: If `clf` does not contain the instance methods necessary for scikit-plot
            instance methods.
    """
    required_methods = ['fit', 'score', 'predict']

    for method in required_methods:
        if not hasattr(clf, method):
            raise ValueError('"{}" is not in clf. Did you pass a classifier instance?'.format(method))

    optional_methods = ['predict_proba']

    for method in optional_methods:
        if not hasattr(clf, method):
            warnings.warn('{} not in clf. Some plots may not be possible to generate.'.format(method))

    clf.plot_learning_curve = types.MethodType(plot_learning_curve, clf)
    return clf


def plot_learning_curve(clf, X, y, title='Learning Curve', cv=None, train_sizes=None, n_jobs=1,
                        ax=None):
    """Generates a plot of the train and test learning curves fora given classifier.

    Args:
        clf: Object type that implements "fit" and "predict" methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features), optional):
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
            learning curve. If None, np.linspace(.1, 1.0, 5) is used.

        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.

        ax (:object:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.
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
