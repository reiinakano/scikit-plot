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
  plot_silhouette,
)

def convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


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