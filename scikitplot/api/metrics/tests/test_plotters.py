import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris as load_data
from sklearn.decomposition import PCA

import scikitplot as skplt


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
        ax = skplt.api.plotters.plot_pca_component_variance(clf, target_explained_variance=0)
        ax = skplt.api.plotters.plot_pca_component_variance(clf, target_explained_variance=0.5)
        ax = skplt.api.plotters.plot_pca_component_variance(clf, target_explained_variance=1)
        ax = skplt.api.plotters.plot_pca_component_variance(clf, target_explained_variance=1.5)

    def test_fitted(self):
        np.random.seed(0)
        clf = PCA()
        self.assertRaises(TypeError, skplt.api.plotters.plot_pca_component_variance, clf)

    def test_ax(self):
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)
        fig, ax = plt.subplots(1, 1)
        out_ax = skplt.api.plotters.plot_pca_component_variance(clf)
        assert ax is not out_ax
        out_ax = skplt.api.plotters.plot_pca_component_variance(clf, ax=ax)
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
        out_ax = skplt.api.plotters.plot_pca_2d_projection(clf, self.X, self.y)
        assert ax is not out_ax
        out_ax =skplt.api.plotters.plot_pca_2d_projection(clf, self.X, self.y, ax=ax)
        assert ax is out_ax

    def test_cmap(self):
        np.random.seed(0)
        clf = PCA()
        clf.fit(self.X)
        fig, ax = plt.subplots(1, 1)
        ax = skplt.api.plotters.plot_pca_2d_projection(clf, self.X, self.y, cmap='Spectral')
        ax = skplt.api.plotters.plot_pca_2d_projection(clf, self.X, self.y, cmap=plt.cm.Spectral)


class TestValidateLabels(unittest.TestCase):

    def test_valid_equal(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B", "C"]
        arg_name = "true_labels"

        actual = skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)
        self.assertEqual(actual, None)

    def test_valid_subset(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B"]
        arg_name = "true_labels"

        actual = skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)
        self.assertEqual(actual, None)

    def test_invalid_one_duplicate(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B", "B"]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)

        msg = "The following duplicate labels were passed into true_labels: B"
        self.assertEqual(msg, str(context.exception))

    def test_invalid_two_duplicates(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B", "A", "B"]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)

        msg = "The following duplicate labels were passed into true_labels: A, B"
        self.assertEqual(msg, str(context.exception))

    def test_invalid_one_missing(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "B", "D"]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)

        msg = "The following labels were passed into true_labels, but were not found in labels: D"
        self.assertEqual(msg, str(context.exception))

    def test_invalid_two_missing(self):
        known_labels = ["A", "B", "C"]
        passed_labels = ["A", "E", "B", "D"]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)

        msg = "The following labels were passed into true_labels, but were not found in labels: E, D"
        self.assertEqual(msg, str(context.exception))

    def test_numerical_labels(self):
        known_labels = [0, 1, 2]
        passed_labels = [0, 2]
        arg_name = "true_labels"

        actual = skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)
        self.assertEqual(actual, None)

    def test_invalid_duplicate_numerical_labels(self):
        known_labels = [0, 1, 2]
        passed_labels = [0, 2, 2]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)

        msg = "The following duplicate labels were passed into true_labels: 2"
        self.assertEqual(msg, str(context.exception))

    def test_invalid_missing_numerical_labels(self):
        known_labels = [0, 1, 2]
        passed_labels = [0, 2, 3]
        arg_name = "true_labels"

        with self.assertRaises(ValueError) as context:
            skplt.api.plotters.validate_labels(known_labels, passed_labels, arg_name)

        msg = "The following labels were passed into true_labels, but were not found in labels: 3"
        self.assertEqual(msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
