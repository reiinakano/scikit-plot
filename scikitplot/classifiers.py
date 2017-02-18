from __future__ import absolute_import, division, print_function, unicode_literals
import six
import warnings
import types
import matplotlib.pyplot as plt
import numpy as np
from scikitplot import plotters
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scikitplot.plotters import plot_feature_importances


def classifier_factory(clf):
    """Takes a scikit-learn classifier instance and embeds scikit-plot instance methods in it.

    Args:
        clf: Scikit-learn classifier instance

    Returns:
        The same scikit-learn classifier instance passed in **clf** with embedded scikit-plot instance methods.

    Raises:
        ValueError: If **clf** does not contain the instance methods necessary for scikit-plot
            instance methods.
    """
    required_methods = ['fit', 'score', 'predict']

    for method in required_methods:
        if not hasattr(clf, method):
            raise TypeError('"{}" is not in clf. Did you pass a classifier instance?'.format(method))

    optional_methods = ['predict_proba']

    for method in optional_methods:
        if not hasattr(clf, method):
            warnings.warn('{} not in clf. Some plots may not be possible to generate.'.format(method))

    additional_methods = {
        'plot_learning_curve': plot_learning_curve,
        'plot_confusion_matrix': plot_confusion_matrix,
        'plot_roc_curve': plot_roc_curve,
        'plot_ks_statistic': plot_ks_statistic,
        'plot_precision_recall_curve': plot_precision_recall_curve,
        'plot_feature_importances': plot_feature_importances
    }

    for key, fn in six.iteritems(additional_methods):
        if hasattr(clf, key):
            warnings.warn('"{}" method already in clf. '
                          'Overriding anyway. This may result in unintended behavior.'.format(key))
        setattr(clf, key, types.MethodType(fn, clf))
    return clf


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


def plot_confusion_matrix(clf, X, y, title=None, normalize=False, do_cv=True, cv=None,
                          shuffle=True, ax=None):
    """Generates the confusion matrix for a given classifier and dataset.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict`` methods.

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

        shuffle (bool, optional): Used when do_cv is set to True. Determines whether to shuffle the
            training data before splitting using cross-validation. Default set to True.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
        >>> rf = classifier_factory(RandomForestClassifier())
        >>> rf.plot_learning_curve(X, y, normalize=True)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_confusion_matrix.png
           :align: center
           :alt: Confusion matrix
    """

    if not do_cv:
        y_pred = clf.predict(X)
        y_true = y

    else:
        if cv is None:
            cv = StratifiedKFold(shuffle=shuffle)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=shuffle)
        else:
            pass

        clf_clone = clone(clf)

        preds_list = []
        trues_list = []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_clone.fit(X_train, y_train)
            preds = clf_clone.predict(X_test)
            preds_list.append(preds)
            trues_list.append(y_test)
        y_pred = np.concatenate(preds_list)
        y_true = np.concatenate(trues_list)

    ax = plotters.plot_confusion_matrix(y_true=y_true, y_pred=y_pred,
                                        title=title, normalize=normalize, ax=ax)

    return ax


def plot_roc_curve(clf, X, y, title='ROC Curves', do_split=True,
                   test_split_ratio=0.33, random_state=None, ax=None):
    """Generates the ROC curves for a given classifier and dataset.

    Args:
        clf: Classifier instance that implements "fit" and "predict_proba" methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        title (string, optional): Title of the generated plot. Defaults to "ROC Curves".

        do_split (bool, optional): If True, the dataset is split into training and testing sets.
            The classifier is trained on the training set and the ROC curve is plotted using the
            performance of the classifier on the testing set. If False, the ROC curves are generated
            without splitting the dataset or training the classifier. This assumes that the
            classifier has already been called with its `fit` method beforehand.

        test_split_ratio (float, optional): Used when do_split is set to True. Determines the
            proportion of the entire dataset to use in the testing split. Default is set to 0.33.

        random_state (int :class:`RandomState`): Pseudo-random number generator state used
            for random sampling.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
            >>> nb = classifier_factory(GaussianNB())
            >>> nb.plot_roc_curve(X, y, random_state=1)
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
            >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. Cannot calculate ROC Curve.')

    if not do_split:
        probas = clf.predict_proba(X)
        y_true = y

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio,
                                                            stratify=y, random_state=random_state)
        clf_clone = clone(clf)
        probas = clf_clone.fit(X_train, y_train).predict_proba(X_test)
        y_true = y_test

    # Compute ROC curve and ROC area for each class
    ax = plotters.plot_roc_curve(y_true=y_true, y_probas=probas, title=title, ax=ax)
    return ax


def plot_ks_statistic(clf, X, y, title='KS Statistic Plot', do_split=True,
                      test_split_ratio=0.33, random_state=None, ax=None):
    """Generates the KS Statistic plot for a given classifier and dataset.

    Args:
        clf: Classifier instance that implements "fit" and "predict_proba" methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        title (string, optional): Title of the generated plot. Defaults to "KS Statistic Plot".

        do_split (bool, optional): If True, the dataset is split into training and testing sets.
            The classifier is trained on the training set and the KS curves are plotted using the
            performance of the classifier on the testing set. If False, the KS curves are generated
            without splitting the dataset or training the classifier. This assumes that the
            classifier has already been called with its `fit` method beforehand.

        test_split_ratio (float, optional): Used when do_split is set to True. Determines the
            proportion of the entire dataset to use in the testing split. Default is set to 0.33.

        random_state (int :class:`RandomState`): Pseudo-random number generator state used
            for random sampling.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
            >>> lr = classifier_factory(LogisticRegression())
            >>> lr.plot_ks_statistic(X, y, random_state=1)
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
            >>> plt.show()

        .. image:: _static/examples/plot_ks_statistic.png
           :align: center
           :alt: KS Statistic
    """
    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. Cannot calculate ROC Curve.')

    if not do_split:
        probas = clf.predict_proba(X)
        y_true = y

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio,
                                                            stratify=y, random_state=random_state)
        clf_clone = clone(clf)
        clf_clone.fit(X_train, y_train)
        probas = clf_clone.predict_proba(X_test)
        y_true = y_test

    ax = plotters.plot_ks_statistic(y_true, probas, title=title, ax=ax)

    return ax


def plot_precision_recall_curve(clf, X, y, title='Precision-Recall Curve', do_split=True,
                                test_split_ratio=0.33, random_state=None, ax=None):
    """Generates the Precision-Recall curve for a given classifier and dataset.

    Args:
        clf: Classifier instance that implements "fit" and "predict_proba" methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        title (string, optional): Title of the generated plot. Defaults to "Precision-Recall Curve".

        do_split (bool, optional): If True, the dataset is split into training and testing sets.
            The classifier is trained on the training set and the Precision-Recall curves are
            plotted using the performance of the classifier on the testing set. If False, the
            Precision-Recall curves are generated without splitting the dataset or training the
            classifier. This assumes that the classifier has already been called with its `fit`
            method beforehand.

        test_split_ratio (float, optional): Used when do_split is set to True. Determines the
            proportion of the entire dataset to use in the testing split. Default is set to 0.33.

        random_state (int :class:`RandomState`): Pseudo-random number generator state used
            for random sampling.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.

    Example:
            >>> nb = classifier_factory(GaussianNB())
            >>> nb.plot_precision_recall_curve(X, y, random_state=1)
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
            >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. '
                        'Cannot calculate Precision-Recall Curve.')

    if not do_split:
        probas = clf.predict_proba(X)
        y_true = y

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio,
                                                            stratify=y, random_state=random_state)
        clf_clone = clone(clf)
        probas = clf_clone.fit(X_train, y_train).predict_proba(X_test)
        y_true = y_test

    # Compute Precision-Recall curve and area for each class
    ax = plotters.plot_precision_recall_curve(y_true, probas, title=title, ax=ax)
    return ax
