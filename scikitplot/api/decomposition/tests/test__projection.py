import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris as load_data
from sklearn.decomposition import PCA

from scikitplot.api.decomposition import plot_pca_2d_projection


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

    def test_label_order(self):
        '''
        Plot labels should be in the same order as the classes in the provided y-array
        '''
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)

        # define y such that the first entry is 1
        y = np.copy(self.y)
        y[0] = 1  # load_iris is be default orderer (i.e.: 0 0 0 ... 1 1 1 ... 2 2 2)

        # test with len(y) == X.shape[0] with multiple rows belonging to the same class
        ax = plot_pca_2d_projection(clf, self.X, y, cmap='Spectral')
        legend_labels = ax.get_legend_handles_labels()[1]
        self.assertListEqual(['1', '0', '2'], legend_labels)

        # test with len(y) == #classes with each row belonging to an individual class
        y = list(range(len(y)))
        np.random.shuffle(y)
        ax = plot_pca_2d_projection(clf, self.X, y, cmap='Spectral')
        legend_labels = ax.get_legend_handles_labels()[1]
        self.assertListEqual([str(v) for v in y], legend_labels)