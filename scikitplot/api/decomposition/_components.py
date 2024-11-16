"""
The :mod:`~scikitplot.decomposition` module includes plots built specifically
for scikit-learn estimators that are used for dimensionality reduction
e.g. PCA. You can use your own estimators, but these plots assume specific
properties shared by scikit-learn estimators. The specific requirements are
documented per function.

The Scikit-plots Functional API

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
  absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
  division,         # Changes the division operator `/` to always perform true division.
  print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
  unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scikitplot.utils.validation import (
  validate_plotting_kwargs_decorator,
)

## Define __all__ to specify the public interface of the module,
## not required default all above func
__all__ = [
  'plot_pca_component_variance',
]


@validate_plotting_kwargs_decorator
def plot_pca_component_variance(
  clf, 
  *args,
  target_explained_variance=0.75,
  model_type = None, # 'PCA' or 'LDA'
  ## plotting params
  title='Cumulative Explained Variance Ratio by Principal Components',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium",
  x_tick_rotation=0,
  ## additional params
  **kwargs,
):
    """
    Plots PCA components' explained variance ratios. (new in v0.2.2)

    .. versionadded:: 0.2.2

    Parameters
    ----------
    clf : object
        PCA instance that has the ``explained_variance_ratio_`` attribute.

    title : str, optional, default='Cumulative Explained Variance Ratio by Principal Components'
        Title of the generated plot.

    target_explained_variance : float, optional, default=0.75
        Looks for the minimum number of principal components that satisfies this 
        value and emphasizes it on the plot.

    ax : matplotlib.axes.Axes, optional, default=None
        The axes upon which to plot the curve. If None, a new set of axes is created.

    figsize : tuple of int, optional, default=None
        Tuple denoting figure size of the plot (e.g., (6, 6)).

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title. Use e.g., "small", "medium", "large" or integer-values.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot. Use e.g., "small", "medium", "large" or integer-values.

    x_tick_rotation : int, optional, default=0
        Rotates x-axis tick labels by the specified angle.

        .. versionadded:: 0.3.9

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was drawn.

    Examples
    --------
    .. plot::
       :context: close-figs
       :align: center
       :alt: PCA Components Variances
    
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> pca = PCA(random_state=0).fit(X)
        >>> skplt.decomposition.plot_pca_component_variance(pca, target_explained_variance=0.95);
        
    .. plot::
       :context: close-figs
       :align: center
       :alt: LDA Components Variances
    
        >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> clf = LinearDiscriminantAnalysis().fit(X, y)
        >>> skplt.decomposition.plot_pca_component_variance(clf);
    """
    ##################################################################
    ## Preprocessing
    ##################################################################
    # Proceed with your preprocess logic here
    # from numpy.linalg import eig
    if hasattr(clf, 'explained_variance_ratio_'):
        exp_var_ratio = clf.explained_variance_ratio_  
    else: #not hasattr(clf, 'explained_variance_ratio_'):
        raise TypeError('"clf" does not have explained_variance_ratio_ '
                        'attribute. Has the PCA been fitted?')

    exp_var_ratio_padded = np.concatenate(([0], exp_var_ratio))
    cumulative_sum_ratios = np.cumsum(exp_var_ratio_padded)
    size = len(cumulative_sum_ratios)

    # Magic code for figuring out closest value to target_explained_variance
    idx = np.searchsorted(cumulative_sum_ratios, target_explained_variance)-1

    # PCA
    if hasattr(clf, 'components_'):
        model_type = 'Principal' #'PCA'
    # LDA Scalings (Eigenvectors for transformation) (similar to PCA)
    elif hasattr(clf, 'scalings_'):
        model_type = 'LDA'
    else:
        pass
    
    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    # Proceed with your plotting logic here
    ax.plot(range(0, size), cumulative_sum_ratios, '*-')
    if idx < len(cumulative_sum_ratios):
        ax.plot(
            idx, cumulative_sum_ratios[idx],
            # fmt = '[marker][line][color]'
            marker='o', ls=':', color='r',
            label='{0:0.2f} for first {1} components\n({2:0.2f} cut-off threshold)'.format(
                cumulative_sum_ratios[idx],
                idx,
                target_explained_variance
            ),
            markersize=3,
            markeredgewidth=3,
        )
        ax.axhline(y=cumulative_sum_ratios[idx], linestyle=':', color='r', lw=1)
        ax.text(x=0.5, y=cumulative_sum_ratios[idx]+0.03,
                s=f'{target_explained_variance:0.2f} cut-off threshold',
                color = 'black', fontsize=text_fontsize
               )
        
    ax.step(range(0, size), cumulative_sum_ratios, where='mid', label='Cumulative Exp Var Ratio')
    ax.bar( range(0, size), exp_var_ratio_padded, alpha=0.5, align='center', label='Individual Exp Var Ratio')

    # Set title, labels, and formatting
    ax.set_title(title.replace('Principal', model_type), fontsize=title_fontsize)
    ax.set_xlabel(
        f'First n {model_type} Components', 
        fontsize=text_fontsize
    )
    ax.set_ylabel(
        f'Cumulative Explained Variance Ratio\nof First n {model_type} Components',
        fontsize=text_fontsize
    )
    ax.tick_params(labelsize=text_fontsize)
    ax.tick_params(axis='x', rotation=x_tick_rotation)
    
    ax.set_xlim([-0.02, ax.get_xlim()[1]])
    # ax.set_ylim([-0.01, 1.01])

    # Define the desired number of ticks
    num_ticks = 11
    # Set x-axis ticks and labels
    ax.xaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    ax.yaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    # ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    # ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    
    # Enable grid and display legend
    ax.grid(True)    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc='lower right',
            fontsize=text_fontsize,
            title=f'Components',
            alignment='left',
        )
    # plt.tight_layout()
    return ax