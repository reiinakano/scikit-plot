from __future__ import absolute_import
import unittest
import scikitplot.plotters as skplt
from sklearn.datasets import load_iris as load_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class TestPlotPCAComponentVariance(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_target_explained_variance(self):
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)
        ax = skplt.plot_pca_component_variance(clf, target_explained_variance=0)
        ax = skplt.plot_pca_component_variance(clf, target_explained_variance=0.5)
        ax = skplt.plot_pca_component_variance(clf, target_explained_variance=1)
        ax = skplt.plot_pca_component_variance(clf, target_explained_variance=1.5)

    def test_fitted(self):
        np.random.seed(0)
        clf = PCA()
        self.assertRaises(TypeError, skplt.plot_pca_component_variance, clf)

    def test_ax(self):
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = skplt.plot_pca_component_variance(clf)
        assert ax is not out_ax
        out_ax =skplt.plot_pca_component_variance(clf, ax=ax)
        assert ax is out_ax


class TestPlotPCA2DProjection(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_ax(self):
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = skplt.plot_pca_2d_projection(clf, self.X, self.y)
        assert ax is not out_ax
        out_ax =skplt.plot_pca_2d_projection(clf, self.X, self.y, ax=ax)
        assert ax is out_ax


if __name__ == '__main__':
    unittest.main()
