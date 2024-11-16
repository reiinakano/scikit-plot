import copy
import numpy

from .probscale import _minimal_norm
from . import validate
from . import algo


## Define __all__ to specify the public interface of the module,
# not required default all above func
__all__ = [
  '_set_prob_limits',
  'probplot',
  'plot_pos',
  'fit_line',
]


def probplot(
    data,
    ax=None,
    plottype="prob",
    dist=None,
    probax="x",
    problabel=None,
    datascale="linear",
    datalabel=None,
    bestfit=False,
    return_best_fit_results=False,
    estimate_ci=False,
    ci_kws=None,
    pp_kws=None,
    scatter_kws=None,
    line_kws=None,
    **fgkwargs
):
    """
    Probability, percentile, and quantile plots.

    Parameters
    ----------
    data : array-like
        1-dimensional data to be plotted

    ax : matplotlib axes, optional
        The Axes on which to plot. If one is not provided, a new Axes
        will be created.

    plottype : string (default = 'prob')
        Type of plot to be created. Options are:

           - 'prob': probability plot
           - 'pp': percentile plot
           - 'qq': quantile plot


    dist : scipy distribution, optional
        A distribution to compute the scale's tick positions. If not
        specified, a standard normal distribution will be used.

    probax : string, optional (default = 'x')
        The axis ('x' or 'y') that will serve as the probability (or
        quantile) axis.

    problabel, datalabel : string, optional
        Axis labels for the probability/quantile and data axes
        respectively.

    datascale : string, optional (default = 'log')
        Scale for the other axis that is not the probability (or
        quantile) axis.

    bestfit : bool, optional (default is False)
        Specifies whether a best-fit line should be added to the plot.

    return_best_fit_results : bool (default is False)
        If True a dictionary of results of is returned along with the
        figure.

    estimate_ci : bool, optional (False)
        Estimate and draw a confidence band around the best-fit line
        using a percentile bootstrap.

    ci_kws : dict, optional
        Dictionary of keyword arguments passed directly to
        ``viz.fit_line`` when computing the best-fit line.

    pp_kws : dict, optional
        Dictionary of keyword arguments passed directly to
        ``viz.plot_pos`` when computing the plotting positions.

    scatter_kws, line_kws : dict, optional
        Dictionary of keyword arguments passed directly to ``ax.plot``
        when drawing the scatter points and best-fit line, respectively.

    Other Parameters
    ----------------
    color : string, optional
        A directly-specified matplotlib color argument for both the
        data series and the best-fit line if drawn. This argument is
        made available for compatibility for the seaborn package and
        is not recommended for general use. Instead colors should be
        specified within ``scatter_kws`` and ``line_kws``.

        .. note::
           Users should not specify this parameter. It is intended to
           only be used by seaborn when operating within a
           ``FacetGrid``.

    label : string, optional
        A directly-specified legend label for the data series. This
        argument is made available for compatibility for the seaborn
        package and is not recommended for general use. Instead the
        data series label should be specified within ``scatter_kws``.

        .. note::
           Users should not specify this parameter. It is intended to
           only be used by seaborn when operating within a
           ``FacetGrid``.


    Returns
    -------
    fig : matplotlib.Figure
        The figure on which the plot was drawn.

    result : dict of linear fit results, optional
        Keys are:

           - q : array of quantiles
           - x, y : arrays of data passed to function
           - xhat, yhat : arrays of modeled data plotted in best-fit line
           - res : array of coefficients of the best-fit line.

    See also
    --------
    viz.plot_pos
    viz.fit_line
    numpy.polyfit
    scipy.stats.probplot
    scipy.stats.mstats.plotting_positions

    Examples
    --------

    Probability plot with the probabilities on the y-axis

    .. plot::
        :context: close-figs

        >>> import numpy; numpy.random.seed(0)
        >>> from matplotlib import pyplot
        >>> from scipy import stats
        >>> import scikitplot.probscale as probscale
        >>> data = numpy.random.normal(loc=5, scale=1.25, size=37)
        >>> fig = probscale.probplot(data, plottype='prob', probax='y',
        ...          problabel='Non-exceedance probability',
        ...          datalabel='Observed values', bestfit=True,
        ...          line_kws=dict(linestyle='--', linewidth=2),
        ...          scatter_kws=dict(marker='o', alpha=0.5))


    Quantile plot with the quantiles on the x-axis

    .. plot::
        :context: close-figs

        >>> import scikitplot.probscale as probscale
        >>> data = numpy.random.normal(loc=5, scale=1.25, size=37)
        >>> fig = probscale.probplot(data, plottype='qq', probax='x',
        ...          problabel='Theoretical Quantiles',
        ...          datalabel='Observed values', bestfit=True,
        ...          line_kws=dict(linestyle='-', linewidth=2),
        ...          scatter_kws=dict(marker='s', alpha=0.5))

    """

    if dist is None:
        dist = _minimal_norm

    # check input values
    fig, ax = validate.axes_object(ax)
    probax = validate.axis_name(probax, "probability axis")
    problabel = validate.axis_label(problabel)
    datalabel = validate.axis_label(datalabel)

    # default values for symbology options
    scatter_kws = validate.other_options(scatter_kws)
    line_kws = validate.other_options(line_kws)
    pp_kws = validate.other_options(pp_kws)

    # check plottype
    plottype = validate.axis_type(plottype)

    # !-- kwarg that only seaborn should use --!
    _color = fgkwargs.get("color", None)
    if _color is not None:
        scatter_kws["color"] = _color
        line_kws["color"] = _color

    # !-- kwarg that only seaborn should use --!
    _label = fgkwargs.get("label", None)
    if _label is not None:
        scatter_kws["label"] = _label

    # !-- kwarg that only seaborn should use --!
    _marker = fgkwargs.get("marker", None)
    if _marker is not None:
        scatter_kws["marker"] = _marker

    # compute the plotting positions and sort the data
    probs, datavals = plot_pos(data, **pp_kws)
    qntls = dist.ppf(probs)

    # determine how the probability values should be expressed
    if plottype == "qq":
        probvals = qntls
    else:
        probvals = probs * 100

    # set up x, y, Axes for probabilities on the x
    if probax == "x":
        x, y = probvals, datavals
        ax.set_xlabel(problabel)
        ax.set_ylabel(datalabel)
        if plottype == "prob":
            ax.set_xscale("prob", dist=dist)
            fitprobs = "x"
        else:
            fitprobs = None
            if plottype == "pp":
                ax.set_xlim(left=0, right=100)

        ax.set_yscale(datascale)
        fitlogs = "y" if datascale == "log" else None

    # setup x, y, Axes for probabilities on the y
    elif probax == "y":
        y, x = probvals, datavals
        ax.set_xlabel(datalabel)
        ax.set_ylabel(problabel)
        if plottype == "prob":
            ax.set_yscale("prob", dist=dist)
            fitprobs = "y"
        else:
            fitprobs = None
            if plottype == "pp":
                ax.set_ylim(bottom=0, top=100)

        ax.set_xscale(datascale)
        fitlogs = "x" if datascale == "log" else None

    # finally plot the data
    linestyle = scatter_kws.pop("linestyle", "none")
    marker = scatter_kws.pop("marker", "o")
    ax.plot(x, y, linestyle=linestyle, marker=marker, **scatter_kws)

    # maybe do a best-fit and plot
    if bestfit:
        xhat, yhat, model = fit_line(
            x,
            y,
            xhat=sorted(x),
            dist=dist,
            fitprobs=fitprobs,
            fitlogs=fitlogs,
            estimate_ci=estimate_ci,
        )
        ax.plot(xhat, yhat, **line_kws)
        if estimate_ci:
            # for alpha, use half of existing or 0.5 * 0.5 = 0.25
            # for zorder, use 1 less than existing or 1 - 1 = 0
            opts = {
                "facecolor": line_kws.get("color", "k"),
                "edgecolor": "None",
                "alpha": line_kws.get("alpha", 0.5) * 0.5,
                "zorder": line_kws.get("zorder", 1) - 1,
                "label": "95% conf. interval",
            }
            ax.fill_between(xhat, y1=model["yhat_hi"], y2=model["yhat_lo"], **opts)
    else:
        xhat, yhat, model = (None, None, None)

    # set the probability axes limits
    if plottype == "prob":
        _set_prob_limits(ax, probax, len(probs))

    # return the figure and maybe results of the best-fit
    if return_best_fit_results:
        results = dict(q=qntls, x=x, y=y, xhat=xhat, yhat=yhat, res=model)
        return fig, results
    else:
        return fig


def plot_pos(data, postype=None, alpha=None, beta=None, exceedance=False):
    """
    Compute the plotting positions for a dataset. Heavily borrows from
    ``scipy.stats.mstats.plotting_positions``.

    A plotting position is defined as: ``(i-alpha)/(n+1-alpha-beta)``
    where:

        - ``i`` is the rank order
        - ``n`` is the size of the dataset
        - ``alpha`` and ``beta`` are parameters used to adjust the
          positions.

    The values of ``alpha`` and ``beta`` can be explicitly set. Typical
    values can also be access via the ``postype`` parameter. Available
    ``postype`` values (alpha, beta) are:

       "type 4" (alpha=0, beta=1)
            Linear interpolation of the empirical CDF.
       "type 5" or "hazen" (alpha=0.5, beta=0.5)
            Piecewise linear interpolation.
       "type 6" or "weibull" (alpha=0, beta=0)
            Weibull plotting positions. Unbiased exceedance probability
            for all distributions. Recommended for hydrologic
            applications.
       "type 7" (alpha=1, beta=1)
            The default values in R. Not recommended with probability
            scales as the min and max data points get plotting positions
            of 0 and 1, respectively, and therefore cannot be shown.
       "type 8" (alpha=1/3, beta=1/3)
            Approximately median-unbiased.
       "type 9" or "blom" (alpha=0.375, beta=0.375)
            Approximately unbiased positions if the data are normally
            distributed.
       "median" (alpha=0.3175, beta=0.3175)
            Median exceedance probabilities for all distributions
            (used in ``scipy.stats.probplot``).
       "apl" or "pwm" (alpha=0.35, beta=0.35)
            Used with probability-weighted moments.
       "cunnane" (alpha=0.4, beta=0.4)
            Nearly unbiased quantiles for normally distributed data.
            This is the default value.
       "gringorten" (alpha=0.44, beta=0.44)
            Used for Gumble distributions.

    Parameters
    ----------
    data : array-like
        The values whose plotting positions need to be computed.

    postype : string, optional (default: "cunnane")

    alpha, beta : float, optional
        Custom plotting position parameters is the options available
        through the `postype` parameter are insufficient.

    exceedance : bool, optional (default: False)
        Toggles "exceedance" vs "non-exceedance" probabilily plots.
        By default, non-exceedance plots are drawn where the plot
        generally slopes from the lower left to the upper right,
        and show the probability that a new observation will be
        less than a given point. By contrast, exceedance plots show
        the probability that a new observation will be greater than
        a given point.

    Returns
    -------
    plot_pos : numpy.array
        The computed plotting positions, sorted.

    data_sorted : numpy.array
        The original data values, sorted.

    References
    ----------
    https://rdrr.io/cran/lmomco/man/pp.html
    http://astrostatistics.psu.edu/su07/R/html/stats/html/quantile.html
    http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.probplot.html
    http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.mstats.plotting_positions.html

    """

    pos_params = {
        "type 4": (0, 1),
        "type 5": (0.5, 0.5),
        "type 6": (0, 0),
        "type 7": (1, 1),
        "type 8": (1.0 / 3.0, 1.0 / 3.0),
        "type 9": (0.375, 0.375),
        "weibull": (0, 0),
        "median": (0.3175, 0.3175),
        "apl": (0.35, 0.35),
        "pwm": (0.35, 0.35),
        "blom": (0.375, 0.375),
        "hazen": (0.5, 0.5),
        "cunnane": (0.4, 0.4),
        "gringorten": (0.44, 0.44),  # Gumble
    }

    postype = "cunnane" if postype is None else postype
    if alpha is None and beta is None:
        alpha, beta = pos_params[postype.lower()]

    data = numpy.asarray(data, dtype=float).flatten()
    n = data.shape[0]
    pos = numpy.empty_like(data)
    pos[n:] = 0

    sorted_index = data.argsort()
    pos[sorted_index[:n]] = (numpy.arange(1.0, n + 1.0) - alpha) / (
        n + 1.0 - alpha - beta
    )

    if exceedance:
        return pos[sorted_index[::-1]], data[sorted_index]

    return pos[sorted_index], data[sorted_index]


def _set_prob_limits(ax, probax, N):
    """Sets the limits of a probability axis based the number of point.

    Parameters
    ----------
    ax : matplotlib Axes
        The Axes object that will be modified.
    N : int
        Maximum number of points for the series plotted on the Axes.
    which : string
        The axis whose ticklabels will be rotated. Valid values are 'x',
        'y', or 'both'.

    Returns
    -------
    None

    """

    fig, ax = validate.axes_object(ax)
    which = validate.axis_name(probax, "probability axis")

    if N <= 5:
        minval = 10
    elif N <= 10:
        minval = 5
    else:
        minval = 10 ** (-1 * numpy.ceil(numpy.log10(N) - 2))

    if which in ["x", "both"]:
        ax.set_xlim(left=minval, right=100 - minval)
    elif which in ["y", "both"]:
        ax.set_ylim(bottom=minval, top=100 - minval)


def fit_line(
    x,
    y,
    xhat=None,
    fitprobs=None,
    fitlogs=None,
    dist=None,
    estimate_ci=False,
    niter=10000,
    alpha=0.05,
):
    """
    Fits a line to x-y data in various forms (linear, log, prob scales).

    Parameters
    ----------
    x, y : array-like
        Independent and dependent data, respectively.

    xhat : array-like, optional
        The values at which ``yhat`` should should be estimated. If
        not provided, falls back to the sorted values of ``x``.

    fitprobs, fitlogs : str, optional.
        Defines how data should be transformed. Valid values are
        'x', 'y', or 'both'. If using ``fitprobs``, variables should
        be expressed as a percentage, i.e.,
        for a probability transform, data will be transformed with
        ``lambda x: dist.ppf(x / 100.)``.
        For a log transform, ``lambda x: numpy.log(x)``.
        Take care to not pass the same value to both ``fitlogs`` and
        ``figprobs`` as both transforms will be applied.

    dist : distribution, optional
        A fully-spec'd scipy.stats distribution-like object
        such that ``dist.ppf`` and ``dist.cdf`` can be called. If not
        provided, defaults to a minimal implementation of
        ``scipy.stats.norm``.

    estimate_ci : bool, optional (False)
        Estimate and draw a confidence band around the best-fit line
        using a percentile bootstrap.

    niter : int, optional (default = 10000)
        Number of bootstrap iterations if ``estimate_ci`` is provided.

    alpha : float, optional (default = 0.05)
        The confidence level of the bootstrap estimate.

    Returns
    -------
    xhat, yhat : numpy arrays
        Linear model estimates of ``x`` and ``y``.
    results : dict
        Dictionary of linear fit results. Keys include:

          - slope
          - intercept
          - yhat_lo (lower confidence interval of the estimated y-vals)
          - yhat_hi (upper confidence interval of the estimated y-vals)

    """

    fitprobs = validate.fit_argument(fitprobs, "fitprobs")
    fitlogs = validate.fit_argument(fitlogs, "fitlogs")

    # maybe set xhat to default values
    if xhat is None:
        xhat = copy.copy(x)

    # maybe set dist to default value
    if dist is None:
        dist = _minimal_norm

    # maybe compute ppf of x
    if fitprobs in ["x", "both"]:
        x = dist.ppf(x / 100.0)
        xhat = dist.ppf(numpy.array(xhat) / 100.0)

    # maybe compute ppf of y
    if fitprobs in ["y", "both"]:
        y = dist.ppf(y / 100.0)

    # maybe compute log of x
    if fitlogs in ["x", "both"]:
        x = numpy.log(x)

    # maybe compute log of y
    if fitlogs in ["y", "both"]:
        y = numpy.log(y)

    yhat, results = algo._fit_simple(x, y, xhat, fitlogs=fitlogs)

    if estimate_ci:
        yhat_lo, yhat_hi = algo._bs_fit(
            x, y, xhat, fitlogs=fitlogs, niter=niter, alpha=alpha
        )
    else:
        yhat_lo, yhat_hi = None, None

    # maybe undo the ppf transform
    if fitprobs in ["y", "both"]:
        yhat = 100.0 * dist.cdf(yhat)
        if yhat_lo is not None:
            yhat_lo = 100.0 * dist.cdf(yhat_lo)
            yhat_hi = 100.0 * dist.cdf(yhat_hi)

    # maybe undo ppf transform
    if fitprobs in ["x", "both"]:
        xhat = 100.0 * dist.cdf(xhat)

    results["yhat_lo"] = yhat_lo
    results["yhat_hi"] = yhat_hi

    return xhat, yhat, results
