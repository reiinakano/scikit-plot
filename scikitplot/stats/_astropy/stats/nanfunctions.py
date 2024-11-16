# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The functions in this module provide faster versions of np.nan*
functions using the optional bottleneck package if it is installed. If
bottleneck is not installed, then the np.nan* functions are used.
"""

from __future__ import annotations

import functools
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import ArrayLike, NDArray

from .funcs import mad_std

from scikitplot._compat.optional_deps import HAS_BOTTLENECK


if HAS_BOTTLENECK:
    import bottleneck
  
    nansum = bottleneck.nansum
    nanmin = bottleneck.nanmin
    nanmax = bottleneck.nanmax
    nanmean = bottleneck.nanmean
    nanmedian = bottleneck.nanmedian
    nanstd = bottleneck.nanstd
    nanvar = bottleneck.nanvar

else:
    nansum = np.nansum
    nanmin = np.nanmin
    nanmax = np.nanmax
    nanmean = np.nanmean
    nanmedian = np.nanmedian
    nanstd = np.nanstd
    nanvar = np.nanvar


def nanmadstd(
    array: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
) -> float | NDArray:
    """mad_std function that ignores NaNs by default."""
    return mad_std(array, axis=axis, ignore_nan=True)