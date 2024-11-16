import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

__all__ = [
  'plot_colortable',
  'plot_overlapping_colors',
]


def plot_colortable(
  colors: dict,
  *,
  ncols: int = 4,
  sort_colors: bool = True,
  display_hex: bool = True,
) -> plt.Figure:
    """
    Display a table of color swatches with corresponding names and optional hex values.

    Parameters
    ----------
    colors : dict
        A dictionary where keys are color names and values are their corresponding 
        hex color codes (e.g., {'red': '#FF0000', 'green': '#00FF00'}).

    ncols : int, optional
        The number of columns in the color table. Default is 4.

    sort_colors : bool, optional
        If True, colors will be sorted by hue, saturation, value, and name before 
        being displayed. Default is True.

    display_hex : bool, optional
        If True, the hex color code will be displayed next to the swatch. Default is True.

    Returns
    -------
    plt.Figure
        A matplotlib figure object containing the color swatch table.

    Notes
    -----
    This function visualizes a list of colors with their corresponding names 
    and optional hex codes in a tabular format. The colors can be sorted 
    based on their HSV values if the `sort_colors` parameter is set to True. 
    The swatch for each color is displayed alongside its name and hex code (if enabled).
    """    
    cell_width = 248
    cell_height = 24
    swatch_width = 48
    margin = 15

    # Sort colors by hue, saturation, value and name.
    if sort_colors:
        try:
            names = sorted(colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
        except:
            names = sorted(colors, key=lambda k: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(colors.get(k)))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 74

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - margin) / height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )
        if display_hex:
            hex_color = colors[name]
            # Pick text colour based on perceived luminance.
            rgba = mcolors.to_rgba_array([hex_color])
            luma = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
            text_color = 'k' if luma[0] > 0.5 else 'w'
            text_args = dict(fontsize=9)
            ax.text(swatch_start_x + 23, y + 3, hex_color,
                    color=text_color, ha='center', **text_args)

    return fig


def plot_overlapping_colors():
    """
    Plot overlapping color names from CSS4 and XKCD color dictionaries using matplotlib.

    This function identifies overlapping color names between the CSS4 and XKCD color collections 
    provided by `matplotlib`. It plots these overlapping colors side by side, with:
    
    - The CSS4 color on the left.
    - The XKCD color on the right.

    If the CSS4 and XKCD colors have identical hex values, the name is displayed in bold. 
    A horizontal line separates each row for better readability, and column headers indicate 
    the source of each color.

    Parameters
    ----------
    None

    Returns
    -------
    plt.Figure
        A matplotlib figure object containing the plotted overlapping colors.

    Notes
    -----
    This function visualizes the overlapping colors between CSS4 and XKCD, making it easier to 
    compare the color representations from both collections. Each color swatch is labeled with 
    its respective name, and the layout adapts to the number of overlapping colors.
    """    
    overlap = {name for name in mcolors.CSS4_COLORS
               if f'xkcd:{name}' in mcolors.XKCD_COLORS}
    
    fig = plt.figure(figsize=[9, 5])
    ax = fig.add_axes([0, 0, 1, 1])
    
    n_groups = 3
    n_rows = len(overlap) // n_groups + 1
    
    for j, color_name in enumerate(sorted(overlap)):
        css4 = mcolors.CSS4_COLORS[color_name]
        xkcd = mcolors.XKCD_COLORS[f'xkcd:{color_name}'].upper()
    
        # Pick text colour based on perceived luminance.
        rgba = mcolors.to_rgba_array([css4, xkcd])
        luma = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
        css4_text_color = 'k' if luma[0] > 0.5 else 'w'
        xkcd_text_color = 'k' if luma[1] > 0.5 else 'w'
    
        col_shift = (j // n_rows) * 3
        y_pos = j % n_rows
        text_args = dict(fontsize=10, weight='bold' if css4 == xkcd else None)
        ax.add_patch(mpatch.Rectangle((0 + col_shift, y_pos), 1, 1, color=css4))
        ax.add_patch(mpatch.Rectangle((1 + col_shift, y_pos), 1, 1, color=xkcd))
        ax.text(0.5 + col_shift, y_pos + .7, css4,
                color=css4_text_color, ha='center', **text_args)
        ax.text(1.5 + col_shift, y_pos + .7, xkcd,
                color=xkcd_text_color, ha='center', **text_args)
        ax.text(2 + col_shift, y_pos + .7, f'  {color_name}', **text_args)
    
    for g in range(n_groups):
        ax.hlines(range(n_rows), 3 * g, 3 * g + 2.8, color='0.7', linewidth=1)
        ax.text(0.5 + 3 * g, -0.3, 'X11/CSS4', ha='center')
        ax.text(1.5 + 3 * g, -0.3, 'xkcd', ha='center')
    
    ax.set_xlim(0, 3 * n_groups)
    ax.set_ylim(n_rows, -1)
    ax.axis('off')
    
    return fig