import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

import matplotlib.pyplot as plt

import warnings

from sklearn.datasets import load_iris as load_data
from sklearn.cluster import KMeans

import scikitplot as skplt


class TestClassifierFactory(unittest.TestCase):

    def setUp(self):
        class Clusterer:
            def __init__(self):
                pass

            def fit(self):
                pass

            def fit_predict(self):
                pass

        class NotClusterer:
            def __init__(self):
                pass

        self.Clusterer = Clusterer
        self.NotClusterer = NotClusterer

    def test_instance_validation(self):

        clf = self.Clusterer()
        skplt._factory_api.clustering_factory(clf)

        not_clf = self.NotClusterer()
        self.assertRaises(TypeError, skplt._factory_api.clustering_factory, not_clf)

    def test_method_insertion(self):

        clf = self.Clusterer()
        skplt._factory_api.clustering_factory(clf)
        assert hasattr(clf, 'plot_silhouette')
        assert hasattr(clf, 'plot_elbow_curve')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            skplt._factory_api.clustering_factory(clf)

            assert len(w) >= 2
            for warning in w[1:]:
                assert issubclass(warning.category, UserWarning)
                assert ' method already in clf. ' \
                       'Overriding anyway. This may ' \
                       'result in unintended behavior.' in str(warning.message)


class TestPlotSilhouette(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_copy(self):
        np.random.seed(0)
        clf = KMeans()
        skplt._factory_api.clustering_factory(clf)
        ax = clf.plot_silhouette(self.X)
        assert not hasattr(clf, "cluster_centers_")
        ax = clf.plot_silhouette(self.X, copy=False)
        assert hasattr(clf, "cluster_centers_")

    def test_cmap(self):
        np.random.seed(0)
        clf = KMeans()
        skplt._factory_api.clustering_factory(clf)
        ax = clf.plot_silhouette(self.X, cmap='Spectral')
        ax = clf.plot_silhouette(self.X, cmap=plt.cm.Spectral)

    def test_ax(self):
        np.random.seed(0)
        clf = KMeans()
        skplt._factory_api.clustering_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_silhouette(self.X)
        assert ax is not out_ax
        out_ax = clf.plot_silhouette(self.X, ax=ax)
        assert ax is out_ax


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
        skplt._factory_api.clustering_factory(clf)
        self.assertRaises(TypeError, clf.plot_elbow_curve, self.X)

    def test_cluster_ranges(self):
        np.random.seed(0)
        clf = KMeans()
        skplt._factory_api.clustering_factory(clf)
        ax = clf.plot_elbow_curve(self.X, cluster_ranges=range(1, 10))
        ax = clf.plot_elbow_curve(self.X)

    def test_ax(self):
        np.random.seed(0)
        clf = KMeans()
        skplt._factory_api.clustering_factory(clf)
        fig, ax = plt.subplots(1, 1)
        out_ax = clf.plot_elbow_curve(self.X)
        assert ax is not out_ax
        out_ax = clf.plot_elbow_curve(self.X, ax=ax)
        assert ax is out_ax

if __name__ == '__main__':
    unittest.main()
