import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes as data_regression
from sklearn.linear_model import Ridge

from scikitplot.api.metrics import (
  plot_residuals_distribution,
)


class TestPlotResidualsDistribution(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X, self.y = data_regression(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]
      
        self.reg = Ridge(alpha=1.0).fit(self.X, self.y)
        self.y_pred = self.reg.predict(self.X)

    def tearDown(self):
        plt.close("all")

    def test_fig(self):
        np.random.seed(0)
        # Create a figure and 3 axes (subplots)
        fig, ax = plt.subplots(1, 3, figsize=(10,5))
      
        # Test without providing ax: the function should create its own figure and axes
        out_fig = plot_residuals_distribution(self.y, self.y_pred)
        # Ensure that the figure returned is not the one created here (since ax was not passed)
        assert fig is not out_fig, "The function should create a new figure when no axes are provided."
      
        # Test with providing ax: the function should use the provided ax
        out_fig = plot_residuals_distribution(self.y, self.y_pred, fig=fig, ax=ax)
        assert fig is out_fig, "Function should use the provided axes."
        # Ensure the new figure has 3 axes (or however many axes the function is supposed to create)
        assert len(out_fig.axes) == 3, "The returned figure should contain 3 axes."

    def test_ax(self):
        np.random.seed(0)
        # Create a figure and 3 axes (subplots)
        fig, ax = plt.subplots(1, 3, figsize=(10,5))
      
        # Test without providing ax: the function should create its own figure and axes
        out_fig = plot_residuals_distribution(self.y, self.y_pred)
        for i, axis in enumerate(ax):
            assert out_fig.axes[i] is not axis, f"The function should create a new axes at index {i}."
          
        # Test with provided ax: the function should use the provided ax
        out_fig = plot_residuals_distribution(self.y, self.y_pred, ax=ax)    # Ensure the axes in the figure match the passed axes
        for i, axis in enumerate(ax):
            assert fig.axes[i] is axis, f"The function should use the provided axes at index {i}."
        
        # Ensure that the figure returned has the same number of axes
        assert len(fig.axes) == 3, "The function should use all 3 axes in the figure."

    def test_cmap(self):
        np.random.seed(0)
        plot_residuals_distribution(self.y, self.y_pred, cmap='nipy_spectral')
        plot_residuals_distribution(self.y, self.y_pred, cmap=plt.cm.nipy_spectral)