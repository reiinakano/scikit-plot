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

from scikitplot.api.kds import (
    plot_cumulative_gain,
    plot_lift,
    plot_ks_statistic,
)


def convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


class TestPlotLift(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_breast_cancer(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_lift(convert_labels_into_string(self.y), probas, class_index=1)

    # def test_two_classes(self):
    #     np.random.seed(0)
    #     # Test this one on Iris (3 classes)
    #     X, y = load_data(return_X_y=True)
    #     clf = LogisticRegression()
    #     clf.fit(X, y)
    #     probas = clf.predict_proba(X)
    #     self.assertRaises(ValueError, plot_lift, y, probas)

    def test_multi_classes(self):
        np.random.seed(0)
        # Test this one on Iris (3 classes)
        X, y = load_data(return_X_y=True)
        clf = LogisticRegression()
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        plot_lift(y, probas)
        # self.assertRaises(ValueError, plot_lift, y, probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
    
        # Case 1: When ax is not passed, plot_cumulative_gain should return a new Axes object
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_lift(self.y, probas)
        assert ax is not out_ax  # They should not be the same because ax was not passed
    
        # Case 2: When ax is passed, plot_cumulative_gain should return the same Axes object
        out_ax = plot_lift(self.y, probas, ax=ax)
        assert ax is out_ax  # They should be the same because ax was passed

    def test_array_like(self):
        plot_lift([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_lift([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_lift(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])

        # # Optionally, you could also add cases for invalid inputs
        # with pytest.raises(ValueError):
        #     plot_cumulative_gain(['a', 'b'], [[0.8, 0.2], [0.2, 0.8]])  # Invalid y_true


class TestPlotCumulativeGain(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_breast_cancer(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_cumulative_gain(convert_labels_into_string(self.y), probas)

    def test_multi_classes(self):
        np.random.seed(0)
        # Test this one on Iris (3 classes)
        X, y = load_data(return_X_y=True)
        clf = LogisticRegression()
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        plot_cumulative_gain(y, probas)
        # self.assertRaises(ValueError, plot_cumulative_gain, y, probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
    
        # Case 1: When ax is not passed, plot_cumulative_gain should return a new Axes object
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_cumulative_gain(self.y, probas)
        assert ax is not out_ax  # They should not be the same because ax was not passed
    
        # Case 2: When ax is passed, plot_cumulative_gain should return the same Axes object
        out_ax = plot_cumulative_gain(self.y, probas, ax=ax)
        assert ax is out_ax  # They should be the same because ax was passed

    def test_array_like(self):
        plot_cumulative_gain([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_cumulative_gain([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_cumulative_gain(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])

        # Optionally, you could also add cases for invalid inputs
        # with pytest.raises(ValueError):
        #     plot_cumulative_gain(['a', 'b'], [[0.8, 0.2], [0.2, 0.8]])  # Invalid y_true


class TestPlotKSStatistic(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_breast_cancer(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, convert_labels_into_string(self.y))
        probas = clf.predict_proba(self.X)
        plot_ks_statistic(convert_labels_into_string(self.y), probas)

    def test_multi_classes(self):
        np.random.seed(0)
        # Test this one on Iris (3 classes)
        X, y = load_data(return_X_y=True)
        clf = LogisticRegression()
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        # self.assertRaises(ValueError, plot_ks_statistic, y, probas)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
    
        # Case 1: When ax is not passed, plot_cumulative_gain should return a new Axes object
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_ks_statistic(self.y, probas)
        assert ax is not out_ax  # They should not be the same because ax was not passed
    
        # Case 2: When ax is passed, plot_cumulative_gain should return the same Axes object
        out_ax = plot_ks_statistic(self.y, probas, ax=ax)
        assert ax is out_ax  # They should be the same because ax was passed

    def test_array_like(self):
        plot_ks_statistic([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        plot_ks_statistic([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
        plot_ks_statistic(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])

        # Optionally, you could also add cases for invalid inputs
        # with pytest.raises(ValueError):
        #     plot_cumulative_gain(['a', 'b'], [[0.8, 0.2], [0.2, 0.8]])  # Invalid y_true