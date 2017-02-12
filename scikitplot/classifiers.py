from __future__ import absolute_import, division, print_function, unicode_literals
import six
import warnings
import types
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scikitplot.helpers import binary_ks_curve


def classifier_factory(clf):
    """Takes a scikit-learn classifier instance and embeds scikit-plot instance methods in it.

    Args:
        clf: Scikit-learn classifier instance

    Returns:
        The same scikit-learn classifier instance with embedded scikit-plot instance methods.

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
        'plot_precision_recall_curve': plot_precision_recall_curve
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

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
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
                          shuffle=True, ax=None):
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

        shuffle (bool, optional): Used when do_cv is set to True. Determines whether to shuffle the
            training data before splitting using cross-validation. Default set to True.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
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
            cv = StratifiedKFold(shuffle=shuffle)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=shuffle)
        else:
            pass

        clf_clone = clone(clf)

        cms = []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_clone.fit(X_train, y_train)
            preds = clf_clone.predict(X_test)
            cms.append(confusion_matrix(y_test, preds))

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


def plot_roc_curve(clf, X, y, title='ROC Curves', do_split=True,
                   test_split_ratio=0.33, random_state=None, ax=None):
    """Generates the ROC curves for a given classifier and dataset.

    Args:
        clf: Object type that implements "fit" and "predict_proba" methods.

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
    """
    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. Cannot calculate ROC Curve.')

    if not do_split:
        classes = clf.classes_
        probas = clf.predict_proba(X)
        y_true = y

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio,
                                                            stratify=y, random_state=random_state)
        clf_clone = clone(clf)
        probas = clf_clone.fit(X_train, y_train).predict_proba(X_test)
        classes = clf_clone.classes_
        y_true = y_test

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true, probas[:, i], pos_label=classes[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in fpr:
        i += 1
        micro_key += str(i)

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    fpr[micro_key], tpr[micro_key], _ = roc_curve(y_true.ravel(), probas.ravel())
    roc_auc[micro_key] = auc(fpr[micro_key], tpr[micro_key])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    macro_key = 'macro'
    i = 0
    while macro_key in fpr:
        i += 1
        macro_key += str(i)
    fpr[macro_key] = all_fpr
    tpr[macro_key] = mean_tpr
    roc_auc[macro_key] = auc(fpr[macro_key], tpr[macro_key])

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title(title)

    for i in range(len(classes)):
        ax.plot(fpr[i], tpr[i], lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(classes[i], roc_auc[i]))

    ax.plot(fpr[micro_key], tpr[micro_key],
            label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc[micro_key]),
            color='deeppink', linestyle=':', linewidth=4)
    ax.plot(fpr[macro_key], tpr[macro_key],
            label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc[macro_key]),
            color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    return ax


def plot_ks_statistic(clf, X, y, title='KS Statistic Plot', do_split=True,
                      test_split_ratio=0.33, random_state=None, ax=None):
    """Generates the KS Statistic plot for a given classifier and dataset.

    Args:
        clf: Object type that implements "fit" and "predict_proba" methods.

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
    """
    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. Cannot calculate ROC Curve.')

    if not do_split:
        if len(clf.classes_) != 2:
            raise ValueError('Cannot calculate KS statistic for data with '
                             '{} category/ies'.format(len(clf.classes_)))
        probas = clf.predict_proba(X)
        y_true = y

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio,
                                                            stratify=y, random_state=random_state)
        clf_clone = clone(clf)
        clf_clone.fit(X_train, y_train)
        if len(clf_clone.classes_) != 2:
            raise ValueError('Cannot calculate KS statistic for data with '
                             '{} category/ies'.format(len(clf_clone.classes_)))
        probas = clf_clone.predict_proba(X_test)
        y_true = y_test

    # Compute KS Statistic curves
    thresholds, pct1, pct2, ks_statistic, \
        max_distance_at, classes = binary_ks_curve(y_true, probas[:, 1].ravel())

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title(title)

    ax.plot(thresholds, pct1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(thresholds, pct2, lw=3, label='Class {}'.format(classes[1]))
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(max_distance_at, *sorted([pct1[idx], pct2[idx]]),
               label='KS Statistic: {:.3f} at {:.3f}'.format(ks_statistic, max_distance_at),
               linestyle=':', lw=3, color='black')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Percentage below threshold')
    ax.legend(loc='lower right')

    return ax


def plot_precision_recall_curve(clf, X, y, title='Precision-Recall Curve', do_split=True,
                                test_split_ratio=0.33, random_state=None, ax=None):
    """Generates the Precision-Recall curve for a given classifier and dataset.

    Args:
        clf: Object type that implements "fit" and "predict_proba" methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification.

        title (string, optional): Title of the generated plot. Defaults to "Precision-Recall Curve".

        do_split (bool, optional): If True, the dataset is split into training and testing sets.
            The classifier is trained on the training set and the Precision-Recall curves are
            plotted using the performance of the classifier on the testing set. If False, the
            Precision-Recall curves are generatedwithout splitting the dataset or training the
            classifier. This assumes that the classifier has already been called with its `fit`
            method beforehand.

        test_split_ratio (float, optional): Used when do_split is set to True. Determines the
            proportion of the entire dataset to use in the testing split. Default is set to 0.33.

        random_state (int :class:`RandomState`): Pseudo-random number generator state used
            for random sampling.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to plot
            the learning curve. If None, the plot is drawn on a new set of axes.
    """
    if not hasattr(clf, 'predict_proba'):
        raise TypeError('"predict_proba" method not in classifier. '
                        'Cannot calculate Precision-Recall Curve.')

    if not do_split:
        classes = clf.classes_
        probas = clf.predict_proba(X)
        y_true = y

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio,
                                                            stratify=y, random_state=random_state)
        clf_clone = clone(clf)
        probas = clf_clone.fit(X_train, y_train).predict_proba(X_test)
        classes = clf_clone.classes_
        y_true = y_test

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_true, probas[:, i],
                                                            pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i], probas[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(y_true.ravel(),
                                                                        probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas, average='micro')

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title(title)
    for i in range(len(classes)):
        ax.plot(recall[i], precision[i], lw=2,
                label='Precision-recall curve of class {0} '
                      '(area = {1:0.3f})'.format(classes[i], average_precision[i]))
    ax.plot(recall[micro_key], precision[micro_key], lw=2, color='gold',
            label='micro-average Precision-recall curve '
                  '(area = {0:0.3f})'.format(average_precision[micro_key]))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='best')
    return ax
