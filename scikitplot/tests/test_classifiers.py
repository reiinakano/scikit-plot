from __future__ import absolute_import
import unittest
import scikitplot
import warnings
from sklearn.datasets import load_iris as load_data
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import numpy as np
import matplotlib.pyplot as plt


class TestClassifierFactory(unittest.TestCase):

    def setUp(self):
        class Classifier:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

            def score(self):
                pass

            def predict_proba(self):
                pass

        class PartialClassifier:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

            def score(self):
                pass

        class NotClassifier:
            def __init__(self):
                pass

        self.Classifier = Classifier
        self.PartialClassifier = PartialClassifier
        self.NotClassifier = NotClassifier

    def test_instance_validation(self):

        clf = self.Classifier()
        scikitplot.classifier_factory(clf)

        not_clf = self.NotClassifier()
        self.assertRaises(TypeError, scikitplot.classifier_factory, not_clf)

        partial_clf = self.PartialClassifier()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            scikitplot.classifier_factory(partial_clf)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert " not in clf. Some plots may not be possible to generate." in str(w[-1].message)

    def test_method_insertion(self):

        clf = self.Classifier()
        scikitplot.classifier_factory(clf)
        assert hasattr(clf, 'plot_learning_curve')
        assert hasattr(clf, 'plot_confusion_matrix')
        assert hasattr(clf, 'plot_roc_curve')
        assert hasattr(clf, 'plot_ks_statistic')
        assert hasattr(clf, 'plot_precision_recall_curve')
        assert hasattr(clf, 'plot_feature_importances')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            scikitplot.classifier_factory(clf)

            assert len(w) == 6
            for warning in w:
                assert issubclass(warning.category, UserWarning)
                assert ' method already in clf. ' \
                       'Overriding anyway. This may ' \
                       'result in unintended behavior.' in str(warning.message)


class TestPlotLearningCurve(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def test_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_learning_curve(self.X, self.y)
        ax = clf.plot_learning_curve(self.X, self.y, cv=5)

    def test_train_sizes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_learning_curve(self.X, self.y, train_sizes=np.linspace(0.1, 1.0, 8))

    def test_n_jobs(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_learning_curve(self.X, self.y, n_jobs=-1)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_learning_curve(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_learning_curve(self.X, self.y, ax=ax)
        assert ax is out_ax


class TestPlotConfusionMatrix(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def test_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y)
        ax = clf.plot_confusion_matrix(self.X, self.y, cv=5)

    def test_normalize(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y, normalize=True)
        ax = clf.plot_confusion_matrix(self.X, self.y, normalize=False)

    def test_do_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y)
        self.assertRaises(NotFittedError, clf.plot_confusion_matrix, self.X, self.y, do_cv=False)

    def test_shuffle(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y, shuffle=True)
        ax = clf.plot_confusion_matrix(self.X, self.y, shuffle=False)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_confusion_matrix(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_confusion_matrix(self.X, self.y, ax=ax)
        assert ax is out_ax


class TestPlotROCCurve(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def test_do_split(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_roc_curve(self.X, self.y)
        self.assertRaises(AttributeError, clf.plot_roc_curve, self.X, self.y,
                          do_split=False)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_roc_curve(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_roc_curve(self.X, self.y, ax=ax)
        assert ax is out_ax


if __name__ == '__main__':
    unittest.main()
