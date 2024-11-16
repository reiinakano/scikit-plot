from matplotlib import pyplot

from .algo import _bs_fit


def axes_object(ax):
    """Checks if a value if an Axes. If None, a new one is created.
    Both the figure and axes are returned (in that order).

    """

    if ax is None:
        ax = pyplot.gca()
        fig = ax.figure
    elif isinstance(ax, pyplot.Axes):
        fig = ax.figure
    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    return fig, ax


def axis_name(axis, axname):
    """
    Checks that an axis name is in ``{'x', 'y'}``. Raises an error on
    an invalid value. Returns the lower case version of valid values.

    """

    valid_args = ["x", "y"]
    if axis.lower() not in valid_args:
        msg = "Invalid value for {} ({}). Must be on of {}."
        raise ValueError(msg.format(axname, axis, valid_args))

    return axis.lower()


def fit_argument(arg, argname):
    """
    Checks that an axis options is in ``{'x', y', 'both', None}``.
    Raises an error on an invalid value. Returns the lower case version
    of valid values.

    """

    valid_args = ["x", "y", "both", None]
    if arg not in valid_args:
        msg = "Invalid value for {} ({}). Must be on of {}."
        raise ValueError(msg.format(argname, arg, valid_args))
    elif arg is not None:
        arg = arg.lower()

    return arg


def axis_type(axtype):
    """
    Checks that a valid axis type is requested.

      - *pp* - percentile axis
      - *qq* - quantile axis
      - *prob* - probability axis

    Raises an error on an invalid value. Returns the lower case version
    of valid values.

    """

    if axtype.lower() not in ["pp", "qq", "prob"]:
        raise ValueError("invalid axtype: {}".format(axtype))
    return axtype.lower()


def axis_label(label):
    """
    Replaces None with an empty string for axis labels.

    """

    return "" if label is None else label


def other_options(options):
    """
    Replaces None with an empty dict for plotting options.

    """

    return dict() if options is None else options.copy()


def estimator(value):
    if value.lower() in ["res", "resid", "resids", "residual", "residuals"]:
        msg = "Bootstrapping the residuals is not ready yet"
        raise NotImplementedError(msg)
    elif value.lower() in ["fit", "values"]:
        est = _bs_fit
    else:
        raise ValueError('estimator must be either "resid" or "fit".')

    return est
