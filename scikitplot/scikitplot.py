from __future__ import absolute_import
import warnings
import types
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


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

    additional_methods = {
        'plot_learning_curve': plot_learning_curve,
        'plot_confusion_matrix': plot_confusion_matrix
    }

    for key, fn in additional_methods.iteritems():
        if hasattr(clf, key):
            warnings.warn('"{}" method already in clf. '
                          'Overriding anyway. This may result in unintended behavior.'.format(key))
        setattr(clf, key, types.MethodType(fn, clf))
    return clf


def plot_learning_curve(clf, X, y, title='Learning Curve', cv=None, train_sizes=None, n_jobs=1,
                        ax=None):
    """Generates a plot of the train and test learning curves for a given classifier.

    Args:
        clf: Object type that implements "fit" and "predict" methods.

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


def plot_confusion_matrix(clf, X, y, title=None, normalize=False, do_cv=True, cv=None,
                          ax=None):
    """Generates the confusion matrix for a given classifier and dataset.

    Args:
        clf: Object type that implements "fit" and "predict" methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        title (string, optional): Title of the generated plot. Defaults to "Confusion Matrix" if
            `normalize` is True. Else, defaults to "Normalized Confusion Matrix.

        normalize (bool, optional): If True, normalizes the confusion matrix before plotting.
            Defaults to False.

        do_cv (bool, optional): If True, the classifier is cross-validated on the dataset using the
            cross-validation strategy in `cv` to generate the confusion matrix. If False, the
            confusion matrix is generated without training or cross-validating the classifier.
            This assumes that the classifier has already been called with its `fit` method beforehand.

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

        ax (:object:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if not do_cv:
        preds = clf.predict(X)
        cm = confusion_matrix(y, preds)
        classes = clf.classes_

    else:
        if cv is None:
            cv = StratifiedKFold(shuffle=True)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True)
        else:
            pass

        clf_clone = clone(clf)

        cms = []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_clone.fit(X_train, y_train)
            preds = clf_clone.predict(X)
            cms.append(confusion_matrix(y, preds))

        classes = clf_clone.classes_
        cm = np.sum(cms, axis=0)

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
