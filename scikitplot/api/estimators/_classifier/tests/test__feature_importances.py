import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris as load_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from scikitplot.api.estimators import plot_feature_importances


def convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


class TestFeatureImportances(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        clf.fit(self.X, convert_labels_into_string(self.y))
        plot_feature_importances(clf)

    def test_feature_importances_in_clf(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        clf.fit(self.X, self.y)
        # self.assertRaises(TypeError, plot_feature_importances, clf)
        plot_feature_importances(clf, feature_names=["a", "b", "c", "d"])

    def test_feature_names(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        clf.fit(self.X, self.y)
        plot_feature_importances(clf, feature_names=["a", "b", "c", "d"])

    def test_threshold(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        clf.fit(self.X, self.y)
        plot_feature_importances(clf, threshold=1e-10)
        plot_feature_importances(clf, threshold=1e-5)
        plot_feature_importances(clf, threshold=1e-2)

    def test_order(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        clf.fit(self.X, self.y)
        plot_feature_importances(clf, order='ascending')
        plot_feature_importances(clf, order='descending')
        plot_feature_importances(clf, order=None)

    # def test_ax(self):
    #     np.random.seed(0)
    #     clf = RandomForestClassifier()
    #     clf.fit(self.X, self.y)
    #     fig, ax = plt.subplots(1, 1)
    #     out_ax = plot_feature_importances(clf)
    #     assert ax is not out_ax
    #     out_ax = plot_feature_importances(clf, ax=ax)
    #     assert ax is out_ax