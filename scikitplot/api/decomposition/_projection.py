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
  # validate_shapes_decorator,
  # validate_y_true_decorator,
  # validate_y_probas_decorator,
  # validate_y_probas_bounds_decorator,
)


## Define __all__ to specify the public interface of the module,
## not required default all above func
__all__ = [
  'plot_pca_2d_projection',
]


@validate_plotting_kwargs_decorator
# @validate_y_true_decorator
def plot_pca_2d_projection(
  clf, 
  X, 
  y,
  *,
  biplot=False, 
  feature_labels=None,
  dimensions=[0, 1], 
  label_dots=False,   
  model_type = None, # 'PCA' or 'LDA'
  ## plotting params
  title='PCA 2-D Projection',
  ax=None,
  fig=None,
  figsize=None,
  title_fontsize="large",
  text_fontsize="medium", 
  cmap='nipy_spectral', # 'Spectral'
  ## additional params
  **kwargs,
):
    """
    Plots the 2-dimensional projection of PCA on a given dataset.

    Parameters
    ----------
    clf : object
        Fitted PCA instance that can ``transform`` given data set into 2 dimensions.

    X : array-like, shape (n_samples, n_features)
        Feature set to project, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features)
        Target relative to X for labeling.

    biplot : bool, optional, default=False
        If True, the function will generate and plot biplots. If False, the
        biplots are not generated.

    feature_labels : array-like, shape (n_features), optional, default=None
        List of labels that represent each feature of X. Its index position
        must also be relative to the features. If None is given, labels will
        be automatically generated for each feature (e.g. "variable1", "variable2",
        "variable3" ...).

    title : str, optional, default='PCA 2-D Projection'
        Title of the generated plot.

    ax : matplotlib.axes.Axes, optional, default=None
        The axes upon which to plot. If None, a new set of axes is created.

    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.

    cmap : str or matplotlib.colors.Colormap, optional, default='viridis'
        Colormap used for plotting the projection. See Matplotlib Colormap
        documentation for available options:
        https://matplotlib.org/users/colormaps.html

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was drawn.

    Examples
    --------
    .. plot::
       :context: close-figs
       :align: center
       :alt: PCA 2D Projection
    
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.datasets import load_iris as data_3_classes
        >>> import scikitplot as skplt
        >>> X, y = data_3_classes(return_X_y=True, as_frame=True)
        >>> pca = PCA(random_state=0).fit(X)
        >>> skplt.decomposition.plot_pca_2d_projection(pca, X, y, biplot=True, feature_labels=X.columns.tolist());
        
    .. plot::
       :context: close-figs
       :align: center
       :alt: LDA 2D Projection
    
        >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        >>> from sklearn.datasets import load_iris as data_3_classes
        >>> import scikitplot as skplt
        >>> X, y = data_3_classes(return_X_y=True, as_frame=True)
        >>> clf = LinearDiscriminantAnalysis().fit(X, y)
        >>> skplt.decomposition.plot_pca_2d_projection(clf, X, y, biplot=True, feature_labels=X.columns.tolist());
    """
    ##################################################################
    ## Preprocessing
    ##################################################################
    # Proceed with your preprocess logic here
    transformed_X = clf.transform(X)
    # Get unique classes from y, preserving order of class occurence in y (pd.unique)
    _, class_indexes = np.unique(np.array(y), return_index=True)
    classes = np.array(y)[np.sort(class_indexes)]
    
    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    # Proceed with your plotting logic here
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(classes)))
    for label, color in zip(classes, colors):
        ax.scatter(transformed_X[y == label, dimensions[0]], transformed_X[y == label, dimensions[1]],
                   alpha=0.8, lw=2, label=label, color=color)

        if label_dots:
            for dot in transformed_X[y == label][:, dimensions]:
                ax.text(*dot, label)

    # PCA
    if hasattr(clf, 'components_'):
        # model_type = 'Principal Components' #'PCA'
        model_type = 'PCA'
        components = clf.components_
    # LDA Scalings (Eigenvectors for transformation) (similar to PCA)
    elif hasattr(clf, 'scalings_'):
        model_type = 'LDA'
        components = clf.scalings_.T
        components = (components - np.min(components)) / (np.max(components)-np.min(components))
    else:
        pass
    if biplot:
        xs = transformed_X[:, dimensions[0]]
        ys = transformed_X[:, dimensions[1]]            
        vectors = np.transpose(components[dimensions, :])
        vectors_scaled = vectors * [xs.max(), ys.max()]
        for i in range(vectors.shape[0]):
            ax.annotate("", xy=(vectors_scaled[i, dimensions[0]], vectors_scaled[i, dimensions[1]]),
                        xycoords='data', xytext=(0, 0), textcoords='data',
                        arrowprops={'arrowstyle': '-|>', 'ec': 'r'})

            ax.text(vectors_scaled[i, 0] * 1.05, vectors_scaled[i, 1] * 1.05,
                    feature_labels[i] if feature_labels else "Variable" + str(i),
                    color='b', fontsize=text_fontsize)

    # Set title, labels, and formatting
    ax.set_title(title.replace('PCA', model_type), fontsize=title_fontsize)
    ax.set_xlabel(f'{model_type} Component {dimensions[0]+1}', fontsize=text_fontsize)
    ax.set_ylabel(f'{model_type} Component {dimensions[1]+1}', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    # ax.set_xlim([-0.02, ax.get_xlim()[1]])
    # ax.set_ylim([-0.01, 1.01])

    # Define the desired number of ticks
    num_ticks = 11
    # Set x-axis ticks and labels
    ax.xaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    ax.yaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    # ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    # ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    
    # Display legend    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc='best',
            shadow=False,
            scatterpoints=1,
            fontsize=text_fontsize,
            title=f'Classes',
            alignment='left',
        )
    # plt.tight_layout()
    return ax