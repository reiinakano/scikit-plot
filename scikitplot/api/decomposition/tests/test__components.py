import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris as load_data
from sklearn.decomposition import PCA

from scikitplot.api.decomposition import plot_pca_component_variance


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