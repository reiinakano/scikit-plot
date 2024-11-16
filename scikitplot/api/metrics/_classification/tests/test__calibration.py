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

from scikitplot.api.metrics import (
  plot_calibration,
)

def convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


class TestPlotCalibration(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_breast_cancer(return_X_y=True, as_frame=False)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]
        self.lr = LogisticRegression(max_iter=int(1e5))
        self.rf = RandomForestClassifier(random_state=8)
        self.svc = LinearSVC()
        self.lr_probas = self.lr.fit(self.X, self.y).predict_proba(self.X)
        self.rf_probas = self.rf.fit(self.X, self.y).predict_proba(self.X)
        self.svc_scores = self.svc.fit(self.X, self.y).\
            decision_function(self.X)

    def tearDown(self):
        plt.close("all")

    def test_decision_function(self):
        plot_calibration(
            self.y,
            [self.lr_probas, self.rf_probas, self.svc_scores],
        )

    def test_plot_calibration(self):
        plot_calibration(
            self.y, 
            [self.lr_probas, self.rf_probas],
        )

    def test_string_classes(self):
        plot_calibration(
            convert_labels_into_string(self.y),
            [self.lr_probas, self.rf_probas],
        )

    def test_cmap(self):
        plot_calibration(
            self.y,
            [self.lr_probas, self.rf_probas],
            cmap='Spectral'
        )
        plot_calibration(
            self.y, 
            [self.lr_probas, self.rf_probas],
            cmap=plt.cm.Spectral
        )

    def test_ax(self):
        plot_calibration(
            self.y,
            [self.lr_probas, self.rf_probas],
        )
        fig, ax = plt.subplots(1, 1)
        out_ax = plot_calibration(
            self.y,
            [self.lr_probas, self.rf_probas],
        )
        assert ax is not out_ax
        out_ax = plot_calibration(
            self.y,
            [self.lr_probas, self.rf_probas],
            ax=ax
        )
        assert ax is out_ax

    def test_array_like(self):
        plot_calibration(
            self.y, 
            [self.lr_probas.tolist(), self.rf_probas.tolist()],
        )
        plot_calibration(
            convert_labels_into_string(self.y),
            [self.lr_probas.tolist(), self.rf_probas.tolist()],
        )

    def test_invalid_probas_list(self):
        self.assertRaises(
            ValueError, 
            plot_calibration,
            self.y, 
            'notalist',
            
        )

    # def test_not_binary(self):
    #     wrong_y = self.y.copy()
    #     wrong_y[-1] = 3
    #     self.assertRaises(
    #         ValueError,
    #         plot_calibration,
    #         wrong_y,
    #         [self.lr_probas, self.rf_probas],
    #     )

    def test_wrong_estimator_names(self):
        self.assertRaises(
            ValueError,
            plot_calibration,
            self.y,
            [self.lr_probas, self.rf_probas],
            estimator_names=['One']
        )

    def test_wrong_probas_shape(self):
        # self.assertRaises(
        #     ValueError,
        #     plot_calibration,
        #     self.y,
        #     [self.lr_probas.reshape(-1), self.rf_probas],
        # )
        self.assertRaises(
            ValueError,
            plot_calibration,
            self.y,
            [np.random.randn(1, 5)],
        )