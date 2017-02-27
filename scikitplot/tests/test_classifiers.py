from __future__ import absolute_import
import unittest
import scikitplot
import warnings
from sklearn.datasets import load_iris as load_data
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
import numpy as np
import matplotlib.pyplot as plt


def convert_labels_into_string(y_true):
    return ["A" if x==0 else x for x in y_true]


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

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_learning_curve(self.X, convert_labels_into_string(self.y))

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

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, convert_labels_into_string(self.y))

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

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_roc_curve(self.X, convert_labels_into_string(self.y))

    def test_predict_proba(self):
        np.random.seed(0)

        class DummyClassifier:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

            def score(self):
                pass

        clf = DummyClassifier()
        scikitplot.classifier_factory(clf)
        self.assertRaises(TypeError, clf.plot_roc_curve, self.X, self.y)

    def test_do_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_roc_curve(self.X, self.y)
        self.assertRaises(AttributeError, clf.plot_roc_curve, self.X, self.y,
                          do_cv=False)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_roc_curve(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_roc_curve(self.X, self.y, ax=ax)
        assert ax is out_ax

    def test_curve_diffs(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax_macro = clf.plot_roc_curve(self.X, self.y, curves='macro')
        ax_micro = clf.plot_roc_curve(self.X, self.y, curves='micro')
        ax_class = clf.plot_roc_curve(self.X, self.y, curves='each_class')
        self.assertNotEqual(ax_macro, ax_micro, ax_class)

    def test_invalid_curve_arg(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        self.assertRaises(ValueError, clf.plot_roc_curve, self.X, self.y,
                          curves='zzz')

class TestPlotKSStatistic(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_breast_cancer(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_ks_statistic(self.X, convert_labels_into_string(self.y))

    def test_predict_proba(self):
        np.random.seed(0)

        class DummyClassifier:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

            def score(self):
                pass

        clf = DummyClassifier()
        scikitplot.classifier_factory(clf)
        self.assertRaises(TypeError, clf.plot_ks_statistic, self.X, self.y)

    def test_two_classes(self):
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        X, y = load_data(return_X_y=True)
        self.assertRaises(ValueError, clf.plot_ks_statistic, X, y)

    def test_do_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_ks_statistic(self.X, self.y)
        self.assertRaises(AttributeError, clf.plot_ks_statistic, self.X, self.y,
                          do_cv=False)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_ks_statistic(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_ks_statistic(self.X, self.y, ax=ax)
        assert ax is out_ax


class TestPlotPrecisionRecall(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_precision_recall_curve(self.X, convert_labels_into_string(self.y))

    def test_predict_proba(self):
        np.random.seed(0)

        class DummyClassifier:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

            def score(self):
                pass

        clf = DummyClassifier()
        scikitplot.classifier_factory(clf)
        self.assertRaises(TypeError, clf.plot_precision_recall_curve, self.X, self.y)

    def test_do_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        ax = clf.plot_precision_recall_curve(self.X, self.y)
        self.assertRaises(AttributeError, clf.plot_precision_recall_curve, self.X, self.y,
                          do_cv=False)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_precision_recall_curve(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_precision_recall_curve(self.X, self.y, ax=ax)
        assert ax is out_ax


class TestFeatureImportances(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        scikitplot.classifier_factory(clf)
        clf.fit(self.X, convert_labels_into_string(self.y))
        ax = clf.plot_feature_importances()

    def test_feature_importances_in_clf(self):
        np.random.seed(0)
        clf = LogisticRegression()
        scikitplot.classifier_factory(clf)
        clf.fit(self.X, self.y)
        self.assertRaises(TypeError, clf.plot_feature_importances)

    def test_feature_names(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        scikitplot.classifier_factory(clf)
        clf.fit(self.X, self.y)
        ax = clf.plot_feature_importances(feature_names=["a", "b", "c", "d"])

    def test_max_num_features(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        scikitplot.classifier_factory(clf)
        clf.fit(self.X, self.y)
        ax = clf.plot_feature_importances(max_num_features=2)
        ax = clf.plot_feature_importances(max_num_features=4)
        ax = clf.plot_feature_importances(max_num_features=6)

    def test_order(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        scikitplot.classifier_factory(clf)
        clf.fit(self.X, self.y)
        ax = clf.plot_feature_importances(order='ascending')
        ax = clf.plot_feature_importances(order='descending')
        ax = clf.plot_feature_importances(order=None)

    def test_ax(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        scikitplot.classifier_factory(clf)
        clf.fit(self.X, self.y)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_feature_importances()
        assert ax is not out_ax
        out_ax = clf.plot_feature_importances(ax=ax)
        assert ax is out_ax


if __name__ == '__main__':
    unittest.main()
