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
  plot_confusion_matrix,
)

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
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, convert_labels_into_string(self.y))
        preds = clf.predict(self.X)
        plot_confusion_matrix(convert_labels_into_string(self.y), preds)

    def test_normalize(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        plot_confusion_matrix(self.y, preds, normalize=True)
        plot_confusion_matrix(self.y, preds, normalize=False)

    def test_labels(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        plot_confusion_matrix(self.y, preds, labels=[0, 1, 2])

    def test_hide_counts(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        plot_confusion_matrix(self.y, preds, hide_counts=True)

    def test_true_pred_labels(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)

        true_labels = [0, 1]
        pred_labels = [0, 2]

        plot_confusion_matrix(self.y, preds,
                              true_labels=true_labels,
                              pred_labels=pred_labels)

    def test_cmap(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        preds = clf.predict(self.X)
        plot_confusion_matrix(self.y, preds, cmap='nipy_spectral')
        plot_confusion_matrix(self.y, preds, cmap=plt.cm.nipy_spectral)

    # def test_ax(self):
    #     np.random.seed(0)
    #     clf = LogisticRegression(max_iter=int(1e5))
    #     clf.fit(self.X, self.y)
    #     preds = clf.predict(self.X)
    #     fig, ax = plt.subplots(1, 1)
    #     out_ax = plot_confusion_matrix(self.y, preds)
    #     assert ax is not out_ax
    #     out_ax = plot_confusion_matrix(self.y, preds, ax=ax)
    #     assert ax is out_ax

    def test_array_like(self):
        plot_confusion_matrix([0, 'a'], ['a', 0])
        plot_confusion_matrix([0, 1], [1, 0])
        plot_confusion_matrix(['b', 'a'], ['a', 'b'])