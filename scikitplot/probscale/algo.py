import numpy


def _make_boot_index(elements, niter):
    """Generate an array of bootstrap sample sets

    Parameters
    ----------
    elements : int
        The number of rows in the original dataset.
    niter : int
        Number of iteration for the bootstrapping.

    Returns
    -------
    index : numpy array
        A collection of random *indices* that can be used to randomly
        sample a dataset ``niter`` times.

    """
    return numpy.random.randint(low=0, high=elements, size=(niter, elements))


def _fit_simple(x, y, xhat, fitlogs=None):
    """
    Simple linear fit of x and y data using ``numpy.polyfit``.

    Parameters
    ----------
    x, y : array-like
    fitlogs : str, optional.
        Defines which data should be log-transformed. Valid values are
        'x', 'y', or 'both'.

    Returns
    -------
    xhat, yhat : array-like
        Estimates of x and y based on the linear fit
    results : dict
        Dictionary of the fit coefficients

    See also
    --------
    numpy.polyfit

    """

    # do the best-fit
    coeffs = numpy.polyfit(x, y, 1)

    results = {"slope": coeffs[0], "intercept": coeffs[1]}

    # estimate y values
    yhat = _estimate_from_fit(
        xhat,
        coeffs[0],
        coeffs[1],
        xlog=fitlogs in ["x", "both"],
        ylog=fitlogs in ["y", "both"],
    )

    return yhat, results


def _bs_fit(x, y, xhat, fitlogs=None, niter=10000, alpha=0.05):
    """
    Percentile method bootstrapping of linear fit of x and y data using
    ``numpy.polyfit``.

    Parameters
    ----------
    x, y : array-like
    fitlogs : str, optional.
        Defines which data should be log-transformed. Valid values are
        'x', 'y', or 'both'.
    niter : int, optional (default is 10000)
        Number of bootstrap iterations to use
    alpha : float, optional
        Confidence level of the estimate.

    Returns
    -------
    xhat, yhat : array-like
        Estimates of x and y based on the linear fit
    results : dict
        Dictionary of the fit coefficients

    See also
    --------
    numpy.polyfit

    """

    index = _make_boot_index(len(x), niter)
    yhat_array = numpy.array(
        [_fit_simple(x[ii], y[ii], xhat, fitlogs=fitlogs)[0] for ii in index]
    )

    percentiles = 100 * numpy.array([alpha * 0.5, 1 - alpha * 0.5])
    yhat_lo, yhat_hi = numpy.percentile(yhat_array, percentiles, axis=0)
    return yhat_lo, yhat_hi


def _estimate_from_fit(xhat, slope, intercept, xlog=False, ylog=False):
    """Estimate the dependent variables of a linear fit given x-data
    and linear parameters.

    Parameters
    ----------
    xhat : numpy array or pandas Series/DataFrame
        The input independent variable of the fit
    slope : float
        Slope of the best-fit line
    intercept : float
        y-intercept of the best-fit line
    xlog, ylog : bool (default = False)
        Toggles whether or not the logs of the x- or y- data should be
        used to perform the regression.

    Returns
    -------
    yhat : numpy array
        Estimate of the dependent variable.

    """

    xhat = numpy.asarray(xhat)
    if ylog:
        if xlog:
            yhat = numpy.exp(intercept) * xhat**slope
        else:
            yhat = numpy.exp(intercept) * numpy.exp(slope) ** xhat

    else:
        if xlog:
            yhat = slope * numpy.log(xhat) + intercept

        else:
            yhat = slope * xhat + intercept

    return yhat
