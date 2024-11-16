import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris as load_data
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from scikitplot.api.metrics import (
  plot_precision_recall_curve,
  plot_precision_recall,
)

def convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


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
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_precision_recall_curve(convert_labels_into_string(self.y), probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_precision_recall_curve(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_precision_recall_curve(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_curve_diffs(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        ax_micro = plot_precision_recall_curve(self.y, probas, curves='micro')
        ax_class = plot_precision_recall_curve(self.y, probas,
                                               curves='each_class')
        self.assertNotEqual(ax_micro, ax_class)

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_precision_recall_curve(self.y, probas, cmap='nipy_spectral')
        plot_precision_recall_curve(self.y, probas, cmap=plt.cm.nipy_spectral)

    def test_invalid_curve_arg(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
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
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_precision_recall(convert_labels_into_string(self.y), probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_precision_recall(self.y, probas)
        assert ax is not out_ax
        out_ax = plot_precision_recall(self.y, probas, ax=ax)
        assert ax is out_ax

    def test_plot_micro(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_precision_recall(self.y, probas, plot_micro=True)
        plot_precision_recall(self.y, probas, plot_micro=False)

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_precision_recall(self.y, probas, cmap='nipy_spectral')
        plot_precision_recall(self.y, probas, cmap=plt.cm.nipy_spectral)

    def test_classes_to_plot(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        plot_precision_recall(self.y, probas, classes_to_plot=[0, 1])
        plot_precision_recall(self.y, probas, classes_to_plot=np.array([0, 1]))

    def test_array_like(self):
        plot_precision_recall([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_precision_recall([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_precision_recall(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])