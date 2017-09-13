from __future__ import absolute_import
import unittest
import numpy as np
from sklearn.datasets import load_iris as load_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from scikitplot.cluster import plot_elbow_curve


class TestPlotElbow(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_n_clusters_in_clf(self):
        np.random.seed(0)

        class DummyClusterer:
            def __init__(self):
                pass

            def fit(self):
                pass

            def fit_predict(self):
                pass

        clf = DummyClusterer()
        self.assertRaises(TypeError, plot_elbow_curve, clf, self.X)

    def test_cluster_ranges(self):
        np.random.seed(0)
        clf = KMeans()
        plot_elbow_curve(clf, self.X, cluster_ranges=range(1, 10))

    def test_ax(self):
        np.random.seed(0)
        clf = KMeans()
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_elbow_curve(clf, self.X)
        assert ax is not out_ax
        out_ax = plot_elbow_curve(clf, self.X, ax=ax)
        assert ax is out_ax

    def test_n_jobs(self):
        np.random.seed(0)
        clf = KMeans()
        plot_elbow_curve(clf, self.X, n_jobs=2)

    def test_show_cluster_time(self):
        np.random.seed(0)
        clf = KMeans()
        plot_elbow_curve(clf, self.X, show_cluster_time=False)
