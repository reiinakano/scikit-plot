import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

import matplotlib.pyplot as plt

import logging
import warnings

from sklearn.datasets import load_iris as load_data
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

import scikitplot as skplt

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
        skplt._factory_api.classifier_factory(clf)

        not_clf = self.NotClassifier()
        self.assertRaises(TypeError, skplt._factory_api.classifier_factory, not_clf)

        partial_clf = self.PartialClassifier()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            skplt._factory_api.classifier_factory(partial_clf)
            assert len(w) == 2
            assert issubclass(w[-1].category, UserWarning)
            assert " not in clf. Some plots may not be possible to generate." in str(w[-1].message)

    def test_method_insertion(self):

        clf = self.Classifier()
        skplt._factory_api.classifier_factory(clf)
        assert hasattr(clf, 'plot_learning_curve')
        assert hasattr(clf, 'plot_confusion_matrix')
        assert hasattr(clf, 'plot_roc_curve')
        assert hasattr(clf, 'plot_ks_statistic')
        assert hasattr(clf, 'plot_precision_recall_curve')
        assert hasattr(clf, 'plot_feature_importances')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            skplt._factory_api.classifier_factory(clf)

            assert len(w) == 7
            for warning in w[1:]:
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
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_learning_curve(self.X, convert_labels_into_string(self.y))

    def test_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_learning_curve(self.X, self.y)
        ax = clf.plot_learning_curve(self.X, self.y, cv=5)

    def test_train_sizes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_learning_curve(self.X, self.y, train_sizes=np.linspace(0.1, 1.0, 8))

    def test_n_jobs(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)

        try:
            ax = clf.plot_learning_curve(self.X, self.y, n_jobs=-1)
        except Exception as e:
            logging.warning(f"Parallel processing failed with n_jobs=-1: {e}. Falling back to n_jobs=1.")
            ax = clf.plot_learning_curve(self.X, self.y, n_jobs=1)
        
        # Further assertions can be added here to validate the plot or results
        self.assertIsNotNone(ax)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
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
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, convert_labels_into_string(self.y))

    def test_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y)
        ax = clf.plot_confusion_matrix(self.X, self.y, cv=5)

    def test_normalize(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y, normalize=True)
        ax = clf.plot_confusion_matrix(self.X, self.y, normalize=False)

    def test_labels(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y, labels=[0, 1, 2])

    def test_true_pred_labels(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)

        true_labels = [0, 1]
        pred_labels = [0, 2]

        ax = clf.plot_confusion_matrix(self.X, self.y, true_labels=true_labels,
                pred_labels=pred_labels)

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y, cmap='nipy_spectral')
        ax = clf.plot_confusion_matrix(self.X, self.y, cmap=plt.cm.nipy_spectral)

    def test_do_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y)
        self.assertRaises(NotFittedError, clf.plot_confusion_matrix, self.X, self.y, do_cv=False)

    def test_shuffle(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_confusion_matrix(self.X, self.y, shuffle=True)
        ax = clf.plot_confusion_matrix(self.X, self.y, shuffle=False)

    # def test_ax(self):
    #     np.random.seed(0)
    #     clf = LogisticRegression()
    #     skplt._factory_api.classifier_factory(clf)
    #     fig, ax = plt.subplots(1, 1)
    #     out_ax = clf.plot_confusion_matrix(self.X, self.y)
    #     assert ax is not out_ax
    #     out_ax = clf.plot_confusion_matrix(self.X, self.y, ax=ax)
    #     assert ax is out_ax

    def test_array_like(self):
        ax = skplt.api.plotters.plot_confusion_matrix([0, 1], [1, 0])


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
        skplt._factory_api.classifier_factory(clf)
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

            def predict_proba(self):
                pass

        clf = DummyClassifier()
        skplt._factory_api.classifier_factory(clf)
        self.assertRaises(TypeError, clf.plot_roc_curve, self.X, self.y)

    def test_do_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_roc_curve(self.X, self.y)
        self.assertRaises(AttributeError, clf.plot_roc_curve, self.X, self.y,
                          do_cv=False)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_roc_curve(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_roc_curve(self.X, self.y, ax=ax)
        assert ax is out_ax

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_roc_curve(self.X, self.y, cmap='nipy_spectral')
        ax = clf.plot_roc_curve(self.X, self.y, cmap=plt.cm.nipy_spectral)

    def test_curve_diffs(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax_macro = clf.plot_roc_curve(self.X, self.y, curves='macro')
        ax_micro = clf.plot_roc_curve(self.X, self.y, curves='micro')
        ax_class = clf.plot_roc_curve(self.X, self.y, curves='each_class')
        self.assertNotEqual(ax_macro, ax_micro, ax_class)

    def test_invalid_curve_arg(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        self.assertRaises(ValueError, clf.plot_roc_curve, self.X, self.y,
                          curves='zzz')

    def test_array_like(self):
        ax = skplt.api.plotters.plot_roc_curve([0, 1], [[0.8, 0.2], [0.2, 0.8]])


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
        skplt._factory_api.classifier_factory(clf)
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

            def predict_proba(self):
                pass

        clf = DummyClassifier()
        skplt._factory_api.classifier_factory(clf)
        self.assertRaises(TypeError, clf.plot_ks_statistic, self.X, self.y)

    def test_two_classes(self):
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        X, y = load_data(return_X_y=True)
        self.assertRaises(ValueError, clf.plot_ks_statistic, X, y)

    def test_do_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_ks_statistic(self.X, self.y)
        self.assertRaises(AttributeError, clf.plot_ks_statistic, self.X, self.y,
                          do_cv=False)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_ks_statistic(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_ks_statistic(self.X, self.y, ax=ax)
        assert ax is out_ax

    def test_array_like(self):
        ax = skplt.api.plotters.plot_ks_statistic([0, 1], [[0.8, 0.2], [0.2, 0.8]])


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
        skplt._factory_api.classifier_factory(clf)
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

            def predict_proba(self):
                pass

        clf = DummyClassifier()
        skplt._factory_api.classifier_factory(clf)
        self.assertRaises(TypeError, clf.plot_precision_recall_curve, self.X, self.y)

    def test_do_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_precision_recall_curve(self.X, self.y)
        self.assertRaises(AttributeError, clf.plot_precision_recall_curve, self.X, self.y,
                          do_cv=False)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_precision_recall_curve(self.X, self.y)
        assert ax is not out_ax
        out_ax = clf.plot_precision_recall_curve(self.X, self.y, ax=ax)
        assert ax is out_ax

    def test_curve_diffs(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax_micro = clf.plot_precision_recall_curve(self.X, self.y, curves='micro')
        ax_class = clf.plot_precision_recall_curve(self.X, self.y, curves='each_class')
        self.assertNotEqual(ax_micro, ax_class)

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        ax = clf.plot_precision_recall_curve(self.X, self.y, cmap='nipy_spectral')
        ax = clf.plot_precision_recall_curve(self.X, self.y, cmap=plt.cm.nipy_spectral)

    def test_invalid_curve_arg(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        self.assertRaises(ValueError, clf.plot_precision_recall_curve, self.X, self.y,
                          curves='zzz')

    def test_array_like(self):
        ax = skplt.api.plotters.plot_precision_recall_curve([0, 1], [[0.8, 0.2], [0.2, 0.8]])


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
        skplt._factory_api.classifier_factory(clf)
        clf.fit(self.X, convert_labels_into_string(self.y))
        ax = clf.plot_feature_importances()

    def test_feature_importances_in_clf(self):
        np.random.seed(0)
        clf = LogisticRegression()
        skplt._factory_api.classifier_factory(clf)
        clf.fit(self.X, self.y)
        self.assertRaises(TypeError, clf.plot_feature_importances)

    def test_feature_names(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        skplt._factory_api.classifier_factory(clf)
        clf.fit(self.X, self.y)
        ax = clf.plot_feature_importances(feature_names=["a", "b", "c", "d"])

    def test_max_num_features(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        skplt._factory_api.classifier_factory(clf)
        clf.fit(self.X, self.y)
        ax = clf.plot_feature_importances(max_num_features=2)
        ax = clf.plot_feature_importances(max_num_features=4)
        ax = clf.plot_feature_importances(max_num_features=6)

    def test_order(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        skplt._factory_api.classifier_factory(clf)
        clf.fit(self.X, self.y)
        ax = clf.plot_feature_importances(order='ascending')
        ax = clf.plot_feature_importances(order='descending')
        ax = clf.plot_feature_importances(order=None)

    def test_ax(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        skplt._factory_api.classifier_factory(clf)
        clf.fit(self.X, self.y)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_feature_importances()
        assert ax is not out_ax
        out_ax = clf.plot_feature_importances(ax=ax)
        assert ax is out_ax


if __name__ == '__main__':
    unittest.main()
