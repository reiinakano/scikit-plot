import numpy
from matplotlib.transforms import Transform


def _mask_out_of_bounds(a):
    """
    Return a Numpy array where all values outside ]0, 1[ are
    replaced with NaNs. If all values are inside ]0, 1[, the original
    array is returned.
    """
    a = numpy.array(a, float)
    mask = (a <= 0.0) | (a >= 1.0)
    if mask.any():
        return numpy.where(mask, numpy.nan, a)
    return a


def _clip_out_of_bounds(a):
    """
    Return a Numpy array where all values outside ]0, 1[ are
    replaced with eps or 1 - eps. If all values are inside ]0, 1[
    the original array is returned. (eps = 1e-300)
    """
    a = numpy.array(a, float)
    a[a <= 0.0] = 1e-300
    a[a >= 1.0] = 1 - 1e-300
    return a


class _ProbTransformMixin(Transform):
    """
    Mixin for MPL axes transform for quantiles/probabilities or
    percentages.

    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, dist, as_pct=True, out_of_bounds="mask"):
        Transform.__init__(self)
        self.dist = dist
        self.as_pct = as_pct
        self.out_of_bounds = out_of_bounds
        if self.as_pct:
            self.factor = 100.0
        else:
            self.factor = 1.0

        if self.out_of_bounds == "mask":
            self._handle_out_of_bounds = _mask_out_of_bounds
        elif self.out_of_bounds == "clip":
            self._handle_out_of_bounds = _clip_out_of_bounds
        else:
            raise ValueError("`out_of_bounds` muse be either 'mask' or 'clip'")


class ProbTransform(_ProbTransformMixin):
    """
    MPL axes transform class to convert quantiles to probabilities
    or percents.

    Parameters
    ----------
    dist : scipy.stats distribution
        The distribution whose ``cdf`` and ``pdf`` methods will set the
        scale of the axis.
    as_pct : bool, optional (True)
        Toggles the formatting of the probabilities associated with the
        tick labels as percentages (0 - 100) or fractions (0 - 1).
    out_of_bounds : string, optional ('mask' or 'clip')
        Determines how data outside the range of valid values is
        handled. The default behavior is to mask the data.
        Alternatively, the data can be clipped to values arbitrarily
        close to the limits of the scale.

    """

    def transform_non_affine(self, prob):
        with numpy.errstate(divide="ignore", invalid="ignore"):
            prob = self._handle_out_of_bounds(numpy.asarray(prob) / self.factor)
            q = self.dist.ppf(prob)
        return q

    def inverted(self):
        return QuantileTransform(
            self.dist, as_pct=self.as_pct, out_of_bounds=self.out_of_bounds
        )


class QuantileTransform(_ProbTransformMixin):
    """
    MPL axes transform class to convert probabilities or percents to
    quantiles.

    Parameters
    ----------
    dist : scipy.stats distribution
        The distribution whose ``cdf`` and ``pdf`` methods will set the
        scale of the axis.
    as_pct : bool, optional (True)
        Toggles the formatting of the probabilities associated with the
        tick labels as percentages (0 - 100) or fractions (0 - 1).
    out_of_bounds : string, optional ('mask' or 'clip')
        Determines how data outside the range of valid values is
        handled. The default behavior is to mask the data.
        Alternatively, the data can be clipped to values arbitrarily
        close to the limits of the scale.

    """

    def transform_non_affine(self, q):
        with numpy.errstate(divide="ignore", invalid="ignore"):
            prob = self.dist.cdf(q) * self.factor
        return prob

    def inverted(self):
        return ProbTransform(
            self.dist, as_pct=self.as_pct, out_of_bounds=self.out_of_bounds
        )
