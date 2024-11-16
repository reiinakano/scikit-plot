import numpy
import warnings
from matplotlib.scale import ScaleBase
from matplotlib.ticker import (
    FixedLocator,
    NullLocator,
    NullFormatter,
    FuncFormatter,
)

from .transforms import ProbTransform
from .formatters import PctFormatter, ProbFormatter


class _minimal_norm(object):
    """
    A basic implementation of a normal distribution, minimally
    API-compliant with scipy.stats.norm

    """

    _A = -(8 * (numpy.pi - 3.0) / (3.0 * numpy.pi * (numpy.pi - 4.0)))

    @classmethod
    def _approx_erf(cls, x):
        """Approximate solution to the error function

        http://en.wikipedia.org/wiki/Error_function

        """

        guts = -(x**2) * (4.0 / numpy.pi + cls._A * x**2) / (1.0 + cls._A * x**2)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in sign")
            return numpy.sign(x) * numpy.sqrt(1.0 - numpy.exp(guts))

    @classmethod
    def _approx_inv_erf(cls, z):
        """Approximate solution to the inverse error function

        http://en.wikipedia.org/wiki/Error_function

        """

        _b = (2 / numpy.pi / cls._A) + (0.5 * numpy.log(1 - z**2))
        _c = numpy.log(1 - z**2) / cls._A
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in sign")
            return numpy.sign(z) * numpy.sqrt(numpy.sqrt(_b**2 - _c) - _b)

    @classmethod
    def ppf(cls, q):
        """Percent point function (inverse of cdf)

        Wikipedia: https://goo.gl/Rtxjme

        """
        return numpy.sqrt(2) * cls._approx_inv_erf(2 * q - 1)

    @classmethod
    def cdf(cls, x):
        """Cumulative density function

        Wikipedia: https://goo.gl/ciUNLx

        """
        return 0.5 * (1 + cls._approx_erf(x / numpy.sqrt(2)))


class ProbScale(ScaleBase):
    """A probability scale for matplotlib Axes.

    Parameters
    ----------
    axis : a matplotlib axis artist
        The axis whose scale will be set.
    dist : scipy.stats probability distribution, optional
        The distribution whose ppf/cdf methods should be used to compute
        the tick positions. By default, a minimal implementation of the
        ``scipy.stats.norm`` class is used so that scipy is not a
        requirement.

    Examples
    --------
    The most basic use:

    .. plot::
        :context: close-figs

        >>> import scikitplot.probscale  # nothing else needed
        >>> from matplotlib import pyplot
        >>> fig, ax = pyplot.subplots(figsize=(4, 7))
        >>> ax.set_ylim(bottom=0.5, top=99.5)
        >>> ax.set_yscale('prob')

    """

    name = "prob"

    def __init__(self, axis, **kwargs):
        self.dist = kwargs.pop("dist", _minimal_norm)
        self.as_pct = kwargs.pop("as_pct", True)
        self.nonpos = kwargs.pop("nonpos", "mask")
        self._transform = ProbTransform(self.dist, as_pct=self.as_pct)

    @classmethod
    def _get_probs(cls, nobs, as_pct):
        """Returns the x-axis labels for a probability plot based on
        the number of observations (`nobs`).
        """
        if as_pct:
            factor = 1.0
        else:
            factor = 100.0

        order = int(numpy.floor(numpy.log10(nobs)))
        base_probs = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

        axis_probs = base_probs.copy()
        for n in range(order):
            if n <= 2:
                lower_fringe = numpy.array([1, 2, 5])
                upper_fringe = numpy.array([5, 8, 9])
            else:
                lower_fringe = numpy.array([1])
                upper_fringe = numpy.array([9])

            new_lower = lower_fringe / 10 ** (n)
            new_upper = upper_fringe / 10 ** (n) + axis_probs.max()
            axis_probs = numpy.hstack([new_lower, axis_probs, new_upper])

        locs = axis_probs / factor
        return locs

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        log scaling.
        """

        axis.set_major_locator(FixedLocator(self._get_probs(1e8, self.as_pct)))
        if self.as_pct:
            axis.set_major_formatter(FuncFormatter(PctFormatter()))
        else:
            axis.set_major_formatter(FuncFormatter(ProbFormatter()))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        return (vmin <= 0.0 and minpos or vmin, vmax <= 0.0 and minpos or vmax)
