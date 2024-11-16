"""Utility Functions for Figure Objects

:py:class:`~matplotlib.figure.Figure`
    Top level :py:class:`~matplotlib.artist.Artist`, which holds all plot elements.
    Many methods are implemented in :py:class:`~matplotlib.figure.FigureBase`.

:py:class:`~matplotlib.figure.SubFigure`
    A logical figure inside a figure, usually added to a figure (or parent :py:class:`~matplotlib.figure.SubFigure`)
    with :py:meth:`~matplotlib.figure.Figure.add_subfigure` or :py:meth:`~matplotlib.figure.Figure.subfigures` methods.

Figures are typically created using pyplot methods :py:func:`~matplotlib.pyplot.figure`,
:py:func:`~matplotlib.pyplot.subplots`, and :py:func:`~matplotlib.pyplot.subplot_mosaic`.

.. plot::

   >>> import matplotlib.pyplot as plt
   >>> fig, ax = plt.subplots(figsize=(2, 2), facecolor='lightskyblue',
   ...                        layout='constrained')
   >>> fig.suptitle('Figure')
   >>> ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')

"""
import functools
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

## Define __all__ to specify the public interface of the module,
## not required default all belove func
__all__ = [
  'combine_and_save_figures', 
]

######################################################################
## combine figure
######################################################################

def combine_and_save_figures(figures, save_path='combined_figures.png', figsize=None, dpi=100, to_save=True):
    """\
    Combine multiple figures into a single image, save it (if specified), and return the combined figure.

    Parameters
    ----------
    figures : tuple of matplotlib.figure.Figure
        Tuple containing the figures to be combined.
        
    save_path : str, optional
        Path where the combined figure image will be saved. 
        Default is 'combined_figure.png'.
        
    figsize : tuple of two int or float, optional
        Size of the combined figure (width, height) in inches. If None, defaults to 
        (12, 3.15 * num_figures), where num_figures is the number of figures to combine.
        Default is None.
        
    dpi : int, optional
        Dots per inch (DPI) for the saved figure. Higher DPI results in better resolution.
        Default is 100.
        
    to_save : bool, optional
        Whether to save the combined figure to a file. If False, the figure is not saved.
        Default is True.

    Returns
    -------
    combined_fig : matplotlib.figure.Figure
        The combined figure containing all the individual figures.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.plot([1, 2, 3], [4, 5, 6])
    >>> ax1.set_title('Figure 1')
    >>> fig2, ax2 = plt.subplots()
    >>> ax2.bar(['A', 'B', 'C'], [3, 7, 2])
    >>> ax2.set_title('Figure 2')
    >>> # Save the combined figure with default figsize
    >>> combined_fig = combine_and_save_figures((fig1, fig2), 'output.png', dpi=150, to_save=True)
    >>> # Combine figures without saving to a file and with custom figsize
    >>> combined_fig = combine_and_save_figures((fig1, fig2), dpi=150, to_save=False, figsize=(14, 7))
    """
    num_figures = len(figures)
    if figsize is None:
        figsize = (12, 3.15 * num_figures)
        
    combined_fig, ax = plt.subplots(num_figures, 1, figsize=figsize, dpi=dpi)
    
    # If only one figure, ax will not be a list, so we make it a list
    if num_figures == 1:
        ax = [ax]

    for i, fig_item in enumerate(figures):
        canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig_item)
        canvas.draw()
        image = canvas.buffer_rgba()
        ax[i].imshow(image)
        ax[i].axis('off')

    # Adjust the layout so there’s no overlap
    combined_fig.tight_layout()

    # Save the combined figure as an image file if to_save is True
    if to_save:
        combined_fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    # Optional: Close the input figures to free up memory
    for fig_item in figures:
        plt.close(fig_item)

    return combined_fig

######################################################################
## 
######################################################################