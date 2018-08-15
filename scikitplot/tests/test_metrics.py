from __future__ import absolute_import
import unittest

from sklearn.datasets import load_iris as load_data
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt

from scikitplot.metrics import plot_confusion_matrix
from scikitplot.metrics import plot_roc_curve
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_ks_statistic
from scikitplot.metrics import plot_precision_recall_curve
from scikitplot.metrics import plot_precision_recall
from scikitplot.metrics import plot_silhouette
from scikitplot.metrics import plot_calibration_curve
from scikitplot.metrics import plot_cumulative_gain
from scikitplot.metrics import plot_lift_curve


def convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


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
        clf.fit(self.X, convert_labels_into_string(self.y))
        preds = clf.predict(self.X)
        plot_confusion_matrix(convert_labels_into_string(self.y), preds)

    def test_normalize(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        plot_confusion_matrix(self.y, preds, normalize=True)
        plot_confusion_matrix(self.y, preds, normalize=False)

    def test_labels(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        plot_confusion_matrix(self.y, preds, labels=[0, 1, 2])

    def test_hide_counts(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        plot_confusion_matrix(self.y, preds, hide_counts=True)

    def test_true_pred_labels(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)

        true_labels = [0, 1]
        pred_labels = [0, 2]

        plot_confusion_matrix(self.y, preds,
                              true_labels=true_labels,
                              pred_labels=pred_labels)

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        plot_confusion_matrix(self.y, preds, cmap='nipy_spectral')
        plot_confusion_matrix(self.y, preds, cmap=plt.cm.nipy_spectral)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_confusion_matrix(self.y, preds)
        assert ax is not out_ax
        out_ax = plot_confusion_matrix(self.y, preds, ax=ax)
        assert ax is out_ax

    def test_array_like(self):
        plot_confusion_matrix([0, 'a'], ['a', 0])
        plot_confusion_matrix([0, 1], [1, 0])
        plot_confusion_matrix(['b', 'a'], ['a', 'b'])


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
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_roc_curve(convert_labels_into_string(self.y), probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_roc_curve(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_roc_curve(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_roc_curve(self.y, probas, cmap='nipy_spectral')
        plot_roc_curve(self.y, probas, cmap=plt.cm.nipy_spectral)

    def test_curve_diffs(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        ax_macro = plot_roc_curve(self.y, probas, curves='macro')
        ax_micro = plot_roc_curve(self.y, probas, curves='micro')
        ax_class = plot_roc_curve(self.y, probas, curves='each_class')
        self.assertNotEqual(ax_macro, ax_micro, ax_class)

    def test_invalid_curve_arg(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        self.assertRaises(ValueError, plot_roc_curve, self.y, probas,
                          curves='zzz')

    def test_array_like(self):
        plot_roc_curve([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_roc_curve([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_roc_curve(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])


class TestPlotROC(unittest.TestCase):
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
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_roc(convert_labels_into_string(self.y), probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_roc(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_roc(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_roc(self.y, probas, cmap='nipy_spectral')
        plot_roc(self.y, probas, cmap=plt.cm.nipy_spectral)

    def test_plot_micro(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_roc(self.y, probas, plot_micro=False)
        plot_roc(self.y, probas, plot_micro=True)

    def test_plot_macro(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_roc(self.y, probas, plot_macro=False)
        plot_roc(self.y, probas, plot_macro=True)

    def test_classes_to_plot(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_roc(self.y, probas, classes_to_plot=[0, 1])
        plot_roc(self.y, probas, classes_to_plot=np.array([0, 1]))

    def test_array_like(self):
        plot_roc([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_roc([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_roc(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])


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
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_ks_statistic(convert_labels_into_string(self.y), probas)

    def test_two_classes(self):
        np.random.seed(0)
        # Test this one on Iris (3 classes)
        X, y = load_data(return_X_y=True)
        clf = LogisticRegression()
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        self.assertRaises(ValueError, plot_ks_statistic, y, probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_ks_statistic(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_ks_statistic(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_array_like(self):
        plot_ks_statistic([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_ks_statistic([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_ks_statistic(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])


class TestPlotPrecisionRecallCurve(unittest.TestCase):
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
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_precision_recall_curve(convert_labels_into_string(self.y), probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_precision_recall_curve(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_precision_recall_curve(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_curve_diffs(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        ax_micro = plot_precision_recall_curve(self.y, probas, curves='micro')
        ax_class = plot_precision_recall_curve(self.y, probas,
                                               curves='each_class')
        self.assertNotEqual(ax_micro, ax_class)

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_precision_recall_curve(self.y, probas, cmap='nipy_spectral')
        plot_precision_recall_curve(self.y, probas, cmap=plt.cm.nipy_spectral)

    def test_invalid_curve_arg(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        self.assertRaises(ValueError, plot_precision_recall_curve, self.y,
                          probas, curves='zzz')

    def test_array_like(self):
        plot_precision_recall_curve([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_precision_recall_curve([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_precision_recall_curve(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])


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
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_precision_recall(convert_labels_into_string(self.y), probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_precision_recall(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_precision_recall(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_plot_micro(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_precision_recall(self.y, probas, plot_micro=True)
        plot_precision_recall(self.y, probas, plot_micro=False)

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_precision_recall(self.y, probas, cmap='nipy_spectral')
        plot_precision_recall(self.y, probas, cmap=plt.cm.nipy_spectral)

    def test_classes_to_plot(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_precision_recall(self.y, probas, classes_to_plot=[0, 1])
        plot_precision_recall(self.y, probas, classes_to_plot=np.array([0, 1]))

    def test_array_like(self):
        plot_precision_recall([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_precision_recall([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_precision_recall(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])


class TestPlotSilhouette(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_plot_silhouette(self):
        np.random.seed(0)
        clf = KMeans()
        cluster_labels = clf.fit_predict(self.X)
        plot_silhouette(self.X, cluster_labels)

    def test_string_classes(self):
        np.random.seed(0)
        clf = KMeans()
        cluster_labels = clf.fit_predict(self.X)
        plot_silhouette(self.X, convert_labels_into_string(cluster_labels))

    def test_cmap(self):
        np.random.seed(0)
        clf = KMeans()
        cluster_labels = clf.fit_predict(self.X)
        plot_silhouette(self.X, cluster_labels, cmap='Spectral')
        plot_silhouette(self.X, cluster_labels, cmap=plt.cm.Spectral)

    def test_ax(self):
        np.random.seed(0)
        clf = KMeans()
        cluster_labels = clf.fit_predict(self.X)
        plot_silhouette(self.X, cluster_labels)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_silhouette(self.X, cluster_labels)
        assert ax is not out_ax
        out_ax = plot_silhouette(self.X, cluster_labels, ax=ax)
        assert ax is out_ax

    def test_array_like(self):
        plot_silhouette(self.X.tolist(), self.y.tolist())
        plot_silhouette(self.X.tolist(), convert_labels_into_string(self.y))


class TestPlotCalibrationCurve(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_breast_cancer(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]
        self.lr = LogisticRegression()
        self.rf = RandomForestClassifier(random_state=8)
        self.svc = LinearSVC()
        self.lr_probas = self.lr.fit(self.X, self.y).predict_proba(self.X)
        self.rf_probas = self.rf.fit(self.X, self.y).predict_proba(self.X)
        self.svc_scores = self.svc.fit(self.X, self.y).\
            decision_function(self.X)

    def tearDown(self):
        plt.close("all")

    def test_decision_function(self):
        plot_calibration_curve(self.y, [self.lr_probas,
                                        self.rf_probas,
                                        self.svc_scores])

    def test_plot_calibration(self):
        plot_calibration_curve(self.y, [self.lr_probas, self.rf_probas])

    def test_string_classes(self):
        plot_calibration_curve(convert_labels_into_string(self.y),
                               [self.lr_probas, self.rf_probas])

    def test_cmap(self):
        plot_calibration_curve(convert_labels_into_string(self.y),
                               [self.lr_probas, self.rf_probas],
                               cmap='Spectral')
        plot_calibration_curve(convert_labels_into_string(self.y),
                               [self.lr_probas, self.rf_probas],
                               cmap=plt.cm.Spectral)

    def test_ax(self):
        plot_calibration_curve(self.y, [self.lr_probas, self.rf_probas])
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_calibration_curve(self.y,
                                        [self.lr_probas, self.rf_probas])
        assert ax is not out_ax
        out_ax = plot_calibration_curve(self.y,
                                        [self.lr_probas, self.rf_probas],
                                        ax=ax)
        assert ax is out_ax

    def test_array_like(self):
        plot_calibration_curve(self.y, [self.lr_probas.tolist(),
                                        self.rf_probas.tolist()])
        plot_calibration_curve(convert_labels_into_string(self.y),
                               [self.lr_probas.tolist(),
                                self.rf_probas.tolist()])

    def test_invalid_probas_list(self):
        self.assertRaises(ValueError, plot_calibration_curve,
                          self.y, 'notalist')

    def test_not_binary(self):
        wrong_y = self.y.copy()
        wrong_y[-1] = 3
        self.assertRaises(ValueError, plot_calibration_curve,
                          wrong_y, [self.lr_probas, self.rf_probas])

    def test_wrong_clf_names(self):
        self.assertRaises(ValueError, plot_calibration_curve,
                          self.y, [self.lr_probas, self.rf_probas],
                          ['One'])

    def test_wrong_probas_shape(self):
        self.assertRaises(ValueError, plot_calibration_curve,
                          self.y, [self.lr_probas.reshape(-1),
                                   self.rf_probas])
        self.assertRaises(ValueError, plot_calibration_curve,
                          self.y, [np.random.randn(1, 5)])


class TestPlotCumulativeGain(unittest.TestCase):
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
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_cumulative_gain(convert_labels_into_string(self.y), probas)

    def test_two_classes(self):
        np.random.seed(0)
        # Test this one on Iris (3 classes)
        X, y = load_data(return_X_y=True)
        clf = LogisticRegression()
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        self.assertRaises(ValueError, plot_cumulative_gain, y, probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_cumulative_gain(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_cumulative_gain(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_array_like(self):
        plot_cumulative_gain([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_cumulative_gain([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_cumulative_gain(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])


class TestPlotLiftCurve(unittest.TestCase):
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
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_lift_curve(convert_labels_into_string(self.y), probas)

    def test_two_classes(self):
        np.random.seed(0)
        # Test this one on Iris (3 classes)
        X, y = load_data(return_X_y=True)
        clf = LogisticRegression()
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        self.assertRaises(ValueError, plot_lift_curve, y, probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_lift_curve(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_lift_curve(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_array_like(self):
        plot_lift_curve([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_lift_curve([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_lift_curve(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])
