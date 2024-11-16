"""
The :mod:`~scikitplot.metrics` module includes plots for machine learning
evaluation metrics e.g. confusion matrix, silhouette scores, etc.

The Scikit-plots Functional API

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
  absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
  division,         # Changes the division operator `/` to always perform true division.
  print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
  unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import warnings
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
  label_binarize, LabelEncoder,
  MinMaxScaler,
)
from sklearn import metrics
# from sklearn.metrics import (
# )
from sklearn.utils import deprecated

# Sigmoid and Softmax functions
import scipy.stats as stats
# import scipy.special as sc
# from scipy.special import (
#   expit as sigmoid,
#   softmax  
# )

# import statsmodels.api as sm
# from statsmodels.genmod.families import Tweedie
# pip install tweedie probscale
# import tweedie
from scikitplot.stats import tweedie

import scikitplot.probscale as probscale
# Q-Q plot with fitted normal distribution
# sm.qqplot
# probscale.probplot
# stats.probplot

from scikitplot.utils.validation import (
  validate_plotting_kwargs_decorator,
)


## Define __all__ to specify the public interface of the module,
# not required default all above func
__all__ = [
  'plot_residuals_distribution',
]


# @validate_plotting_kwargs_decorator
def plot_residuals_distribution(
  ## default params
  y_true,
  y_pred,
  *,
  dist_type='normal',
  var_power=1.5,
  ## plotting params
  title='Precision-Recall AUC Curves',
  ax=None,
  fig=None,
  figsize=(10, 5),
  title_fontsize="large",
  text_fontsize="medium",
  cmap=None,
  show_labels=True,
  digits=4,
  ## additional params
  **kwargs,
):
    """
    Plot residuals and fit various distributions to assess their goodness of fit.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.
    
    y_pred : array-like, shape (n_samples,)
        Estimated targets as returned by a classifier.
    
    dist_type : str, optional, default='normal'
        Type of distribution to fit to the residuals. Options are:
        
        - 'normal': For symmetrically distributed residuals (mean μ, std σ).
        - 'poisson': For count-based residuals or rare events (mean λ).
        - 'gamma': For positive, skewed residuals with a heavy tail (shape k or α, scale θ or β).
        - 'inverse_gaussian': For residuals with a distribution similar to the inverse Gaussian (mean μ, scale λ).
        - 'exponential': For non-negative residuals with a long tail (scale λ).
        - 'lognormal': For positively skewed residuals with a multiplicative effect (shape σ, scale exp(μ)).
        - 'tweedie': For complex data including counts and continuous components. 
            
        The Tweedie distribution can model different types of data based on the variance power (`var_power`):
                
        - var_power = 0: Normal distribution (mean μ, std σ)
        - var_power = 1: Poisson distribution (mean λ)
        - 1 < var_power < 2: Compound Poisson-Gamma distribution
        - var_power = 2: Gamma distribution (shape k, scale θ)
        - var_power = 3: Inverse Gaussian distribution (mean μ, scale λ)
        
    var_power : float or None
        The variance power for the Tweedie distribution, applicable if `dist_type='tweedie'`.
            - Default is 1.5, which means Tweedie-specific plotting.
            - Example values: 1.5 for Compound Poisson-Gamma distribution, 2 for Gamma distribution.
    
    title : str, optional, default='Precision-Recall AUC Curves'
        Title of the generated plot.
    
    ax : list of matplotlib.axes.Axes, optional, default=None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).
        Axes like ``fig.add_subplot(1, 1, 1)`` or ``plt.gca()``
    
    fig : matplotlib.pyplot.figure, optional, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).
    
        .. versionadded:: 0.3.9
    
    figsize : tuple of int, optional, default=(10, 5)
        Size of the figure (width, height) in inches.
    
    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.
    
    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.
    
    cmap : None, str or matplotlib.colors.Colormap, optional, default=None
        Colormap used for plotting.
        Options include 'viridis', 'PiYG', 'plasma', 'inferno', 'nipy_spectral', etc.
        See Matplotlib Colormap documentation for available choices.
        
        - https://matplotlib.org/stable/users/explain/colors/index.html
        - plt.colormaps()
        - plt.get_cmap()  # None == 'viridis'
    
    show_labels : bool, optional, default=True
        Whether to display the legend labels.
    
    digits : int, optional, default=3
        Number of digits for formatting PR AUC values in the plot.
    
        .. versionadded:: 0.3.9
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.
    
    Raises
    ------
    ValueError: If an unsupported distribution type is provided or if `var_power` is invalid.
    
    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Residuals Distribution
    
       >>> import numpy as np; np.random.seed(0)
       >>> from sklearn.datasets import load_diabetes as data_regression
       >>> from sklearn.model_selection import train_test_split
       >>> from sklearn.linear_model import Ridge
       >>> import scikitplot as skplt
       >>> 
       >>> X, y = data_regression(return_X_y=True, as_frame=False)
       >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
       >>> model = Ridge(alpha=1.0).fit(X_train, y_train)
       >>> y_val_pred = model.predict(X_val)
       >>> skplt.metrics.plot_residuals_distribution(
       >>>     y_val, y_val_pred, dist_type='tweedie',
       >>> );
    """
    ##################################################################
    ## Preprocessing
    ##################################################################
    # Proceed with your preprocess logic here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
  
    # Compute residuals (y - y_hat)
    residuals = y_true - y_pred
    
    ##################################################################
    ## Plotting
    ##################################################################
    # Validate the types of ax and fig if they are provided
    if ax is not None and not all(isinstance(ax, mpl.axes.Axes) for ax in ax):
        raise ValueError(
            "Provided ax list must be an instance of matplotlib.axes.Axes"
        )
    if fig is not None and not isinstance(fig, mpl.figure.Figure):
        raise ValueError(
            "Provided fig must be an instance of matplotlib.figure.Figure"
        )
    # Neither ax nor fig is provided.
    # Create a new figure with two subplots,
    # adjusting the width ratios with the specified figsize.
    if ax is None and fig is None:
        fig, ax = plt.subplots(
            nrows=1, ncols=3, 
            figsize=figsize, 
            sharex=False, 
            sharey=False, 
            # gridspec_kw={'width_ratios': [5, 5]}
        )
    # fig is provided but ax is not.
    # Add two subplots (ax) to the provided figure (fig).
    elif ax is None:
        # 111 means a grid of 1 row, 1 column, and we want the first (and only) subplot.
        # fig = plt.gcf()
        ax = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
    # ax is provided (whether fig is provided or not).
    # Use the provided ax for plotting. No new figure or subplot is created.
    else:
        pass        
    # Proceed with your plotting logic here
  
    
    # Histogram of residuals
    ax[0].hist(residuals, bins='auto', edgecolor='k', alpha=0.8, color='dodgerblue')
    ax[0].set_title("Histogram of Residuals")
    ax[0].set_xlabel("Residuals")
    ax[0].set_ylabel("Frequency")

    if dist_type == 'normal':
        # Fit normal distribution
        mean, std = stats.norm.fit(residuals)
        print(f"Fitted mean-mu (μ): {mean:.4f}")
        print(f"Fitted std (σ)    : {std:.4f}")

        # Q-Q plot with fitted normal distribution
        # sm.qqplot
        # probscale.probplot
        stats.probplot(residuals, dist="norm", sparams=(mean, std), rvalue=True, fit=True, plot=ax[1])
        ax[1].set_title(f"Q-Q Plot: Fitted Normal\nμ={mean:.2f}, σ={std:.2f}")

        # Q-Q plot with standard normal distribution
        stats.probplot(residuals, dist=stats.norm, rvalue=True, fit=True, plot=ax[2])
        ax[2].set_title(f"Q-Q Plot: Standard Normal\nmean=0, std=1")

    elif dist_type == 'poisson':
        # Fit Poisson distribution
        lambda_fitted = np.mean(residuals)
        print(f"Fitted lambda (λ): {lambda_fitted:.4f}")

        # Q-Q plot with fitted Poisson distribution
        stats.probplot(residuals, dist=stats.poisson, sparams=(lambda_fitted,), rvalue=True, fit=True, plot=ax[1])
        ax[1].set_title(f"Q-Q Plot: Fitted Poisson\nλ={lambda_fitted:.2f}")

        # Q-Q plot with standard Poisson distribution
        stats.probplot(residuals, dist=stats.poisson, sparams=(1,), rvalue=True, fit=True, plot=ax[2])
        ax[2].set_title(f"Q-Q Plot: Standard Poisson\nλ=1")

    elif dist_type == 'gamma':
        # Shift residuals to ensure all values are positive
        residuals = residuals - np.min(residuals) + 1e-12
        
        # Fit gamma distribution
        shape, loc, scale = stats.gamma.fit(residuals, floc=0)
        print(f"Fitted shape shape-alpha (k or α): {shape:.4f}")
        print(f"Fitted scale scale-beta  (θ or β): {scale:.4f}")

        # Q-Q plot with fitted gamma distribution
        stats.probplot(residuals, dist=stats.gamma, sparams=(shape, loc, scale), rvalue=True, fit=True, plot=ax[1])
        ax[1].set_title(f"Q-Q Plot: Fitted Gamma\nk={shape:.2f}, loc={loc:.2f}, θ={scale:.2f}")

        # Q-Q plot with standard gamma distribution
        stats.probplot(residuals, dist=stats.gamma, sparams=(1, 0, 1), rvalue=True, fit=True, plot=ax[2])
        ax[2].set_title(f"Q-Q Plot: Standard Gamma\nshape=1, location=0, scale=1")

    elif dist_type == 'inverse_gaussian':
        # Fit Inverse Gaussian distribution
        shape, loc, scale = stats.invgauss.fit(residuals)
        print(f"Fitted shape (λ): {shape:.4f}")
        print(f"Fitted location (μ): {loc:.4f}")
        print(f"Fitted scale (λ): {scale:.4f}")

        # Q-Q plot with fitted Inverse Gaussian distribution
        stats.probplot(residuals, dist=stats.invgauss, sparams=(shape, loc, scale), rvalue=True, fit=True, plot=ax[1])
        ax[1].set_title(f"Q-Q Plot: Fitted Inverse Gaussian\nλ={shape:.2f}, μ={loc:.2f}, scale={scale:.2f}")

        # Q-Q plot with standard Inverse Gaussian distribution
        stats.probplot(residuals, dist=stats.invgauss, sparams=(1, 0, 1), rvalue=True, fit=True, plot=ax[2])
        ax[2].set_title(f"Q-Q Plot: Standard Inverse Gaussian\nshape=1, location=0, scale=1")

    elif dist_type == 'tweedie':
        # Fit Tweedie distribution
        mean, std = stats.norm.fit(residuals)
        print(f"Fitted mean-mu (μ): {mean:.4f}")
        print(f"Tweedie var_power :{var_power:.4f}")

        # Q-Q plot with fitted Tweedie distribution
        # Tweedie does not have a direct probplot function, so this may need specialized plotting
        stats.probplot(residuals, dist=tweedie(mu=mean, p=var_power, phi=1.0), sparams=(), rvalue=True, fit=True, plot=ax[1])
        ax[1].set_title(f"Q-Q Plot: Fitted Tweedie\nμ={mean:.2f}, var_power={var_power:.2f}")

        # Q-Q plot with standard Tweedie distribution
        stats.probplot(residuals, dist=tweedie(mu=mean, p=0, phi=1.0), sparams=(), rvalue=True, fit=True, plot=ax[2])
        ax[2].set_title(f"Q-Q Plot: Standard Tweedie\nμ={mean:.2f}, var_power=0")

    elif dist_type == 'exponential':
        # Fit exponential distribution
        scale = np.mean(residuals)
        print(f"Fitted scale (λ): {scale:.4f}")

        # Q-Q plot with fitted exponential distribution
        stats.probplot(residuals, dist=stats.expon, sparams=(scale,), rvalue=True, fit=True, plot=ax[1])
        ax[1].set_title(f"Q-Q Plot: Fitted Exponential\nλ={scale:.2f}")

        # Q-Q plot with standard exponential distribution
        stats.probplot(residuals, dist=stats.expon, sparams=(1,), rvalue=True, fit=True, plot=ax[2])
        ax[2].set_title(f"Q-Q Plot: Standard Exponential\nλ=1")

    elif dist_type == 'lognormal':
        # Shift residuals to ensure all values are positive
        residuals = residuals - np.min(residuals) + 1e-12

        # Fit log-normal distribution
        shape, loc, scale = stats.lognorm.fit(residuals, floc=0)
        print(f"Fitted shape sigma (σ)       : {shape:.4f}")
        print(f"Fitted scale exp(mu) (exp(μ)): {scale:.4f}")

        # Q-Q plot with fitted log-normal distribution
        stats.probplot(residuals, dist=stats.lognorm, sparams=(shape, loc, scale), rvalue=True, fit=True, plot=ax[1])
        ax[1].set_title(f"Q-Q Plot: Fitted Log-Normal\nσ={shape:.2f}, scale={scale:.2f}")

        # Q-Q plot with standard log-normal distribution
        stats.probplot(residuals, dist=stats.lognorm, sparams=(1, 0, 1), rvalue=True, fit=True, plot=ax[2])
        ax[2].set_title(f"Q-Q Plot: Standard Log-Normal\nshape=1, location=0, scale=1")

    else:
        raise ValueError(
            "Unsupported distribution type."
            "Choose from 'normal', 'poisson', 'gamma', 'inverse_gaussian', 'tweedie', 'exponential', or 'lognormal'.")


    # Enable grid and display legend
    for axis in ax:
        # axis.grid(True)
        if show_labels:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(
                    loc='lower left',
                    fontsize=text_fontsize,
                    title='',
                    alignment='left'
                )
        
    # equalize the scales
    # plt.axis('equal')
    # ax.set_aspect(aspect='equal', adjustable='box')    

    # Improve layout and show plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig