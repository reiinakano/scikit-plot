from __future__ import absolute_import, division, print_function, \
    unicode_literals
import six
import warnings
import types

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.utils import deprecated

from scikitplot import plotters
from scikitplot.plotters import plot_feature_importances
from scikitplot.plotters import plot_learning_curve


@deprecated('This will be removed in v0.4.0. The Factory '
            'API has been deprecated. Please migrate '
            'existing code into the various new modules '
            'of the Functions API. Please note that the '
            'interface of those functions will likely be '
            'different from that of the Factory API.')
def classifier_factory(clf):
    """Embeds scikit-plot instance methods in an sklearn classifier.

    Args:
        clf: Scikit-learn classifier instance

    Returns:
        The same scikit-learn classifier instance passed in **clf**
        with embedded scikit-plot instance methods.

    Raises:
        ValueError: If **clf** does not contain the instance methods
            necessary for scikit-plot instance methods.
    """
    required_methods = ['fit', 'score', 'predict']

    for method in required_methods:
        if not hasattr(clf, method):
            raise TypeError('"{}" is not in clf. Did you pass a '
                            'classifier instance?'.format(method))

    optional_methods = ['predict_proba']

    for method in optional_methods:
        if not hasattr(clf, method):
            warnings.warn('{} not in clf. Some plots may '
                          'not be possible to generate.'.format(method))

    additional_methods = {
        'plot_learning_curve': plot_learning_curve,
        'plot_confusion_matrix': plot_confusion_matrix_with_cv,
        'plot_roc_curve': plot_roc_curve_with_cv,
        'plot_ks_statistic': plot_ks_statistic_with_cv,
        'plot_precision_recall_curve': plot_precision_recall_curve_with_cv,
        'plot_feature_importances': plot_feature_importances
    }

    for key, fn in six.iteritems(additional_methods):
        if hasattr(clf, key):
            warnings.warn('"{}" method already in clf. '
                          'Overriding anyway. This may '
                          'result in unintended behavior.'.format(key))
        setattr(clf, key, types.MethodType(fn, clf))
    return clf


def plot_confusion_matrix_with_cv(clf, X, y, labels=None, true_labels=None,
                                  pred_labels=None, title=None,
                                  normalize=False, hide_zeros=False,
                                  x_tick_rotation=0, do_cv=True, cv=None,
                                  shuffle=True, random_state=None, ax=None,
                                  figsize=None, cmap='Blues',
                                  title_fontsize="large",
                                  text_fontsize="medium"):
    """Generates the confusion matrix for a given classifier and dataset.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict``
            methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        labels (array-like, shape (n_classes), optional): List of labels to
            index the matrix. This may be used to reorder or select a subset of
            labels. If none is given, those that appear at least once in ``y``
            are used in sorted order.
            (new in v0.2.5)

        true_labels (array-like, optional): The true labels to display.
            If none is given, then all of the labels are used.

        pred_labels (array-like, optional): The predicted labels to display.
            If none is given, then all of the labels are used.

        title (string, optional): Title of the generated plot. Defaults to
            "Confusion Matrix" if normalize` is True. Else, defaults to
            "Normalized Confusion Matrix.

        normalize (bool, optional): If True, normalizes the confusion matrix
            before plotting. Defaults to False.

        hide_zeros (bool, optional): If True, does not plot cells containing a
            value of zero. Defaults to False.

        x_tick_rotation (int, optional): Rotates x-axis tick labels by the
            specified angle. This is useful in cases where there are numerous
            categories and the labels overlap each other.

        do_cv (bool, optional): If True, the classifier is cross-validated on
            the dataset using the cross-validation strategy in `cv` to generate
            the confusion matrix. If False, the confusion matrix is generated
            without training or cross-validating the classifier. This assumes
            that the classifier has already been called with its `fit` method
            beforehand.

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

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

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
        >>> rf = classifier_factory(RandomForestClassifier())
        >>> rf.plot_confusion_matrix(X, y, normalize=True)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_confusion_matrix.png
           :align: center
           :alt: Confusion matrix
    """
    y = np.array(y)

    if not do_cv:
        y_pred = clf.predict(X)
        y_true = y

    else:
        if cv is None:
            cv = StratifiedKFold(shuffle=shuffle, random_state=random_state)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=shuffle,
                                 random_state=random_state)
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
                                        labels=labels, true_labels=true_labels,
                                        pred_labels=pred_labels,
                                        title=title, normalize=normalize,
                                        hide_zeros=hide_zeros,
                                        x_tick_rotation=x_tick_rotation, ax=ax,
                                        figsize=figsize, cmap=cmap,
                                        title_fontsize=title_fontsize,
                                        text_fontsize=text_fontsize)

    return ax


def plot_roc_curve_with_cv(clf, X, y, title='ROC Curves', do_cv=True,
                           cv=None, shuffle=True, random_state=None,
                           curves=('micro', 'macro', 'each_class'),
                           ax=None, figsize=None, cmap='nipy_spectral',
                           title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves for a given classifier and dataset.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict``
            methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        do_cv (bool, optional): If True, the classifier is cross-validated on
            the dataset using the cross-validation strategy in `cv` to generate
            the confusion matrix. If False, the confusion matrix is generated
            without training or cross-validating the classifier. This assumes
            that the classifier has already been called with its `fit` method
            beforehand.

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

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "macro", "each_class")`
            i.e. "micro" for micro-averaged curve, "macro" for macro-averaged
            curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

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
            >>> nb = classifier_factory(GaussianNB())
            >>> nb.plot_roc_curve(X, y, random_state=1)
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
            >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y = np.array(y)

    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. '
                        'Cannot calculate ROC Curve.')

    if not do_cv:
        probas = clf.predict_proba(X)
        y_true = y

    else:
        if cv is None:
            cv = StratifiedKFold(shuffle=shuffle, random_state=random_state)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=shuffle,
                                 random_state=random_state)
        else:
            pass

        clf_clone = clone(clf)

        preds_list = []
        trues_list = []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_clone.fit(X_train, y_train)
            preds = clf_clone.predict_proba(X_test)
            preds_list.append(preds)
            trues_list.append(y_test)
        probas = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(trues_list)

    # Compute ROC curve and ROC area for each class
    ax = plotters.plot_roc_curve(y_true=y_true, y_probas=probas, title=title,
                                 curves=curves, ax=ax, figsize=figsize,
                                 cmap=cmap, title_fontsize=title_fontsize,
                                 text_fontsize=text_fontsize)

    return ax


def plot_ks_statistic_with_cv(clf, X, y, title='KS Statistic Plot',
                              do_cv=True, cv=None, shuffle=True,
                              random_state=None, ax=None, figsize=None,
                              title_fontsize="large", text_fontsize="medium"):
    """Generates the KS Statistic plot for a given classifier and dataset.

    Args:
        clf: Classifier instance that implements "fit" and "predict_proba"
            methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        title (string, optional): Title of the generated plot. Defaults to
            "KS Statistic Plot".

        do_cv (bool, optional): If True, the classifier is cross-validated on
            the dataset using the cross-validation strategy in `cv` to generate
            the confusion matrix. If False, the confusion matrix is generated
            without training or cross-validating the classifier. This assumes
            that the classifier has already been called with its `fit` method
            beforehand.

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

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

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
            >>> lr = classifier_factory(LogisticRegression())
            >>> lr.plot_ks_statistic(X, y, random_state=1)
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
            >>> plt.show()

        .. image:: _static/examples/plot_ks_statistic.png
           :align: center
           :alt: KS Statistic
    """
    y = np.array(y)

    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. '
                        'Cannot calculate ROC Curve.')

    if not do_cv:
        probas = clf.predict_proba(X)
        y_true = y

    else:
        if cv is None:
            cv = StratifiedKFold(shuffle=shuffle, random_state=random_state)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=shuffle,
                                 random_state=random_state)
        else:
            pass

        clf_clone = clone(clf)

        preds_list = []
        trues_list = []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_clone.fit(X_train, y_train)
            preds = clf_clone.predict_proba(X_test)
            preds_list.append(preds)
            trues_list.append(y_test)
        probas = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(trues_list)

    ax = plotters.plot_ks_statistic(y_true, probas, title=title,
                                    ax=ax, figsize=figsize,
                                    title_fontsize=title_fontsize,
                                    text_fontsize=text_fontsize)

    return ax


def plot_precision_recall_curve_with_cv(clf, X, y,
                                        title='Precision-Recall Curve',
                                        do_cv=True, cv=None, shuffle=True,
                                        random_state=None,
                                        curves=('micro', 'each_class'),
                                        ax=None, figsize=None,
                                        cmap='nipy_spectral',
                                        title_fontsize="large",
                                        text_fontsize="medium"):
    """Generates the Precision-Recall curve for a given classifier and dataset.

    Args:
        clf: Classifier instance that implements "fit" and "predict_proba"
            methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        title (string, optional): Title of the generated plot. Defaults to
            "Precision-Recall Curve".

        do_cv (bool, optional): If True, the classifier is cross-validated on
            the dataset using the cross-validation strategy in `cv` to generate
            the confusion matrix. If False, the confusion matrix is generated
            without training or cross-validating the classifier. This assumes
            that the classifier has already been called with its `fit` method
            beforehand.

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

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "each_class")`
            i.e. "micro" for micro-averaged curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

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
            >>> nb = classifier_factory(GaussianNB())
            >>> nb.plot_precision_recall_curve(X, y, random_state=1)
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
            >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    y = np.array(y)

    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. '
                        'Cannot calculate Precision-Recall Curve.')

    if not do_cv:
        probas = clf.predict_proba(X)
        y_true = y

    else:
        if cv is None:
            cv = StratifiedKFold(shuffle=shuffle, random_state=random_state)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=shuffle,
                                 random_state=random_state)
        else:
            pass

        clf_clone = clone(clf)

        preds_list = []
        trues_list = []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_clone.fit(X_train, y_train)
            preds = clf_clone.predict_proba(X_test)
            preds_list.append(preds)
            trues_list.append(y_test)
        probas = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(trues_list)

    # Compute Precision-Recall curve and area for each class
    ax = plotters.plot_precision_recall_curve(y_true, probas, title=title,
                                              curves=curves, ax=ax,
                                              figsize=figsize, cmap=cmap,
                                              title_fontsize=title_fontsize,
                                              text_fontsize=text_fontsize)
    return ax
