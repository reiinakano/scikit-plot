import numpy as numpy
from matplotlib.ticker import Formatter


class _FormatterMixin(Formatter):
    """A mpl-axes formatter mixin class"""

    @classmethod
    def _sig_figs(cls, x, n, expthresh=5, forceint=False):
        """
        Formats a number with the correct number of significant digits.

        Parameters
        ----------
        x : int or float
            The number you want to format.
        n : int
            The number of significant figures it should have.
        expthresh : int, optional (default = 5)
            The absolute value of the order of magnitude at which numbers
            are formatted in exponential notation.
        forceint : bool, optional (default is False)
            If true, simply returns int(x)

        Returns
        -------
        formatted : str
            The formatted number as a string

        Examples
        --------
        >>> _sig_figs(1247.15, 3)
        '1250'
        >>> _sig_figs(1247.15, 7)
        '1247.150'

        """

        # return a string value unaltered
        if isinstance(x, str):
            out = cls._sig_figs(float(x), n, expthresh=expthresh, forceint=forceint)

        elif x == 0.0:
            out = "0"

        # check on the number provided
        elif x is not None and numpy.isfinite(x):
            # check on the _sig_figs
            if n < 1:
                raise ValueError("number of sig figs (n) must be greater " "than zero")

            elif forceint:
                out = "{:,.0f}".format(x)

            # logic to do all of the rounding
            else:
                order = numpy.floor(numpy.log10(numpy.abs(x)))

                if -1.0 * expthresh <= order <= expthresh:
                    decimal_places = int(n - 1 - order)

                    if decimal_places <= 0:
                        out = "{0:,.0f}".format(round(x, decimal_places))

                    else:
                        fmt = "{0:,.%df}" % decimal_places
                        out = fmt.format(x)

                else:
                    decimal_places = n - 1
                    fmt = "{0:.%de}" % decimal_places
                    out = fmt.format(x)

        # with NAs and INFs, just return 'NA'
        else:
            out = "NA"

        return out

    def __call__(self, x, pos=None):
        if x < (10 / self.factor):
            out = self._sig_figs(x, 1)
        elif x <= (99 / self.factor):
            out = self._sig_figs(x, 2)
        else:
            order = numpy.ceil(numpy.round(numpy.abs(numpy.log10(self.top - x)), 6))
            out = self._sig_figs(x, order + self.offset)

        return "{}".format(out)


class PctFormatter(_FormatterMixin):
    """
    Formatter class for MPL axes to display probabilities as percentages.

    Examples
    --------
    >>> from probscale import formatters
    >>> fmt = formatters.PctFormatter()
    >>> fmt(0.2)
    '0.2'
    >>> fmt(10)
    '10'
    >>> fmt(99.999)
    '99.999'

    """

    factor = 1.0
    offset = 2
    top = 100


class ProbFormatter(_FormatterMixin):
    """
    Formatter class for MPL axes to display probabilities as decimals.

    Examples
    --------
    >>> from probscale import formatters
    >>> fmt = formatters.ProbFormatter()
    >>> fmt(0.01)
    '0.01'
    >>> fmt(0.2)
    '0.20'
    >>> try:
    ...    fmt(10.5)
    ... except(ValueError):
    ...     print('formatter out of bounds')
    formatter out of bounds
    """

    factor = 100.0
    offset = 0
    top = 1
