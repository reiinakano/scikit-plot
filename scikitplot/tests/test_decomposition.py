from __future__ import absolute_import
import unittest

from sklearn.datasets import load_iris as load_data
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

from scikitplot.decomposition import plot_pca_component_variance
from scikitplot.decomposition import plot_pca_2d_projection


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
        plot_pca_component_variance(clf, target_explained_variance=0)
        plot_pca_component_variance(clf, target_explained_variance=0.5)
        plot_pca_component_variance(clf, target_explained_variance=1)
        plot_pca_component_variance(clf, target_explained_variance=1.5)

    def test_fitted(self):
        np.random.seed(0)
        clf = PCA()
        self.assertRaises(TypeError, plot_pca_component_variance, clf)

    def test_ax(self):
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_pca_component_variance(clf)
        assert ax is not out_ax
        out_ax = plot_pca_component_variance(clf, ax=ax)
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
        out_ax = plot_pca_2d_projection(clf, self.X, self.y)
        assert ax is not out_ax
        out_ax = plot_pca_2d_projection(clf, self.X, self.y, ax=ax)
        assert ax is out_ax

    def test_cmap(self):
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)
        plot_pca_2d_projection(clf, self.X, self.y, cmap='Spectral')
        plot_pca_2d_projection(clf, self.X, self.y, cmap=plt.cm.Spectral)

    def test_biplot(self):
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)
        ax = plot_pca_2d_projection(clf, self.X, self.y, biplot=True,
                                    feature_labels=load_data().feature_names)
