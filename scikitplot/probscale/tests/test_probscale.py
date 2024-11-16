import os
import warnings

import numpy
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

import pytest

# import scikitplot.probscale as probscale
from .. import __init__ as probscale
from ..probscale import _minimal_norm


# special toloerance for Github Action CI
TOLERANCE = int(os.environ.get("MPL_IMGCOMP_TOLERANCE", 15))
BASELINE_DIR = "baseline_images/test_probscale"


@pytest.fixture
def mn():
    return _minimal_norm()


@pytest.fixture
def mn_input():
    x = numpy.array(
        [
            0.331,
            0.742,
            0.067,
            0.826,
            0.357,
            0.089,
            0.754,
            0.342,
            0.762,
            0.658,
            0.239,
            0.910,
        ]
    )
    return x


def test_minimal_norm_A(mn):
    known__A = 0.1400122886866665
    assert abs(mn._A - known__A) < 0.0000001


def test_minimal_norm__approx_erf(mn, mn_input):
    known_erf = numpy.array(
        [
            0.36029027,
            0.70598131,
            0.07548843,
            0.75724986,
            0.38635283,
            0.10016122,
            0.71371964,
            0.37137355,
            0.71880142,
            0.64791492,
            0.26463458,
            0.80188283,
        ]
    )

    diff = mn._approx_erf(mn_input) - known_erf
    assert numpy.all(numpy.abs(diff) < 0.001)


def test_minimal_norm__approx_inv_erf(mn, mn_input):
    diff = mn._approx_inv_erf(mn._approx_erf(mn_input)) - mn_input
    assert numpy.all(numpy.abs(diff) < 0.00001)


def test_minimal_norm_ppf(mn, mn_input):
    known_ppf = numpy.array(
        [
            -0.43715354,
            0.64952360,
            -1.49851307,
            0.93847570,
            -0.36648929,
            -1.34693863,
            0.68713129,
            -0.40701088,
            +0.71275076,
            0.40701088,
            -0.70952297,
            1.34075503,
        ]
    )
    diff = mn.ppf(mn_input) - known_ppf
    assert numpy.all(numpy.abs(diff) < 0.001)


def test_minimal_norm_cdf(mn, mn_input):
    known_cdf = numpy.array(
        [
            0.62967776,
            0.77095633,
            0.52670915,
            0.79559795,
            0.63945410,
            0.53545904,
            0.77457539,
            0.63382455,
            0.77697000,
            0.74473093,
            0.59444721,
            0.81858875,
        ]
    )
    diff = mn.cdf(mn_input) - known_cdf
    assert numpy.all(numpy.abs(diff) < 0.001)


def test_sign_with_nan_no_warning(mn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = mn._approx_erf(numpy.nan)
    assert numpy.isnan(res)


def test_sign_with_nan_no_warning_inv(mn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = mn._approx_inv_erf(numpy.nan)
    assert numpy.isnan(res)


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=TOLERANCE, remove_text=True
)
def test_the_scale_default():
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_yscale("prob")
    ax.set_ylim(0.01, 99.99)
    fig.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
def test_the_scale_not_as_pct():
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_yscale("prob", as_pct=False)
    ax.set_ylim(0.02, 0.98)
    return fig


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=TOLERANCE, remove_text=True
)
@pytest.mark.skipif(stats is None, reason="scipy not installed")
def test_the_scale_beta():
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_yscale("prob", as_pct=True, dist=stats.beta(3, 2))
    ax.set_ylim(1, 99)
    fig.tight_layout()
    return fig
