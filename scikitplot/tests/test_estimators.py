from __future__ import absolute_import
import unittest
from sklearn.datasets import load_iris as load_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

from scikitplot.estimators import plot_feature_importances
from scikitplot.estimators import plot_learning_curve


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
        clf = LogisticRegression()
        clf.fit(self.X, self.y)
        self.assertRaises(TypeError, plot_feature_importances, clf)

    def test_feature_names(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        clf.fit(self.X, self.y)
        plot_feature_importances(clf, feature_names=["a", "b", "c", "d"])

    def test_max_num_features(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        clf.fit(self.X, self.y)
        plot_feature_importances(clf, max_num_features=2)
        plot_feature_importances(clf, max_num_features=4)
        plot_feature_importances(clf, max_num_features=6)

    def test_order(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        clf.fit(self.X, self.y)
        plot_feature_importances(clf, order='ascending')
        plot_feature_importances(clf, order='descending')
        plot_feature_importances(clf, order=None)

    def test_ax(self):
        np.random.seed(0)
        clf = RandomForestClassifier()
        clf.fit(self.X, self.y)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_feature_importances(clf)
        assert ax is not out_ax
        out_ax = plot_feature_importances(clf, ax=ax)
        assert ax is out_ax


class TestPlotLearningCurve(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_string_classes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        plot_learning_curve(clf, self.X, convert_labels_into_string(self.y))

    def test_cv(self):
        np.random.seed(0)
        clf = LogisticRegression()
        plot_learning_curve(clf, self.X, self.y)
        plot_learning_curve(clf, self.X, self.y, cv=5)

    def test_train_sizes(self):
        np.random.seed(0)
        clf = LogisticRegression()
        plot_learning_curve(clf, self.X, self.y,
                            train_sizes=np.linspace(0.1, 1.0, 8))

    def test_n_jobs(self):
        np.random.seed(0)
        clf = LogisticRegression()
        plot_learning_curve(clf, self.X, self.y, n_jobs=-1)

    def test_random_state_and_shuffle(self):
        np.random.seed(0)
        clf = LogisticRegression()
        plot_learning_curve(clf, self.X, self.y, random_state=1, shuffle=True)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression()
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_learning_curve(clf, self.X, self.y)
        assert ax is not out_ax
        out_ax = plot_learning_curve(clf, self.X, self.y, ax=ax)
        assert ax is out_ax
