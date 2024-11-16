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

from scikitplot.api.estimators import plot_learning_curve


def convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


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
        clf = LogisticRegression(max_iter=int(1e5))
        plot_learning_curve(clf, self.X, convert_labels_into_string(self.y))

    def test_cv(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        plot_learning_curve(clf, self.X, self.y)
        plot_learning_curve(clf, self.X, self.y, cv=5)

    def test_train_sizes(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        plot_learning_curve(clf, self.X, self.y,
                            train_sizes=np.linspace(0.1, 1.0, 8))

    def test_n_jobs(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        plot_learning_curve(clf, self.X, self.y, n_jobs=-1)

    def test_random_state_and_shuffle(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        plot_learning_curve(clf, self.X, self.y, random_state=1, shuffle=True)

    def test_ax(self):
        np.random.seed(0)
        clf = LogisticRegression(max_iter=int(1e5))
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_learning_curve(clf, self.X, self.y)
        assert ax is not out_ax
        out_ax = plot_learning_curve(clf, self.X, self.y, ax=ax)
        assert ax is out_ax