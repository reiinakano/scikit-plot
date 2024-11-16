import numpy

import pytest
import numpy.testing as nptest

from .. import algo


def test__make_boot_index():
    result = algo._make_boot_index(5, 5000)
    assert result.shape == (5000, 5)
    assert result.min() == 0
    assert result.max() == 4


@pytest.fixture
def plot_data():
    data = numpy.array(
        [
            3.113,
            3.606,
            4.046,
            4.046,
            4.710,
            6.140,
            6.978,
            2.000,
            4.200,
            4.620,
            5.570,
            5.660,
            5.860,
            6.650,
            6.780,
            6.790,
            7.500,
            7.500,
            7.500,
            8.630,
            8.710,
            8.990,
            9.850,
            10.820,
            11.250,
            11.250,
            12.200,
            14.920,
            16.770,
            17.810,
            19.160,
            19.190,
            19.640,
            20.180,
            22.970,
        ]
    )
    return data


@pytest.mark.parametrize(
    ("fitlogs", "known_yhat"),
    [
        (None, numpy.array([0.7887, 3.8946, 7.0005, 10.1065, 13.2124, 16.3183])),
        ("x", numpy.array([0.2711, 1.2784, 1.5988, 1.7953, 1.9373, 2.0487])),
        (
            "y",
            numpy.array(
                [2.2006e00, 4.9139e01, 1.0972e03, 2.4501e04, 5.4711e05, 1.2217e07]
            ),
        ),
        ("both", numpy.array([1.3114, 3.5908, 4.9472, 6.0211, 6.9402, 7.7577])),
    ],
)
def test__fit_simple(plot_data, fitlogs, known_yhat):
    x = numpy.arange(1, len(plot_data) + 1)
    known_results = {"slope": 0.5177, "intercept": 0.2711}
    xhat = x[::6]
    yhat, results = algo._fit_simple(x, plot_data, xhat, fitlogs=fitlogs)
    assert abs(results["intercept"] - known_results["intercept"]) < 0.0001
    assert abs(results["slope"] - known_results["slope"]) < 0.0001
    nptest.assert_allclose(yhat, known_yhat, rtol=0.0001)


@pytest.mark.parametrize(
    ("fitlogs", "known_lo", "known_hi"),
    [
        (
            None,
            numpy.array([-0.7944, 2.7051, 6.1974, 9.2612, 11.9382, 14.4290]),
            numpy.array([2.1447, 4.8360, 7.7140, 10.8646, 14.1014, 17.4432]),
        ),
        (
            "x",
            numpy.array([-1.4098, -0.2210, 0.1387, 0.3585, 0.5147, 0.6417]),
            numpy.array([1.7067, 2.5661, 2.8468, 3.0169, 3.1400, 3.2341]),
        ),
        (
            "y",
            numpy.array(
                [4.5187e-01, 1.4956e01, 4.9145e02, 1.0522e04, 1.5299e05, 1.8468e06]
            ),
            numpy.array(
                [8.5396e00, 1.2596e02, 2.2396e03, 5.2290e04, 1.3310e06, 3.7627e07]
            ),
        ),
        (
            "both",
            numpy.array([0.2442, 0.8017, 1.1488, 1.4312, 1.6731, 1.8997]),
            numpy.array([5.5107, 13.0148, 17.232, 20.4285, 23.1035, 25.3843]),
        ),
    ],
)
def test__bs_fit(plot_data, fitlogs, known_lo, known_hi):
    numpy.random.seed(0)
    x = numpy.arange(1, len(plot_data) + 1)
    xhat = x[::6]
    yhat_lo, yhat_hi = algo._bs_fit(x, plot_data, xhat, fitlogs=fitlogs, niter=1000)

    nptest.assert_allclose(yhat_lo, known_lo, rtol=0.001)
    nptest.assert_allclose(yhat_hi, known_hi, rtol=0.001)


class Test__estimate_from_fit(object):
    def setup_method(self):
        self.x = numpy.arange(1, 11, 0.5)
        self.slope = 2
        self.intercept = 3.5

        self.known_ylinlin = numpy.array(
            [
                5.5,
                6.5,
                7.5,
                8.5,
                9.5,
                10.5,
                11.5,
                12.5,
                13.5,
                14.5,
                15.5,
                16.5,
                17.5,
                18.5,
                19.5,
                20.5,
                21.5,
                22.5,
                23.5,
                24.5,
            ]
        )

        self.known_yloglin = numpy.array(
            [
                3.50000000,
                4.31093022,
                4.88629436,
                5.33258146,
                5.69722458,
                6.00552594,
                6.27258872,
                6.50815479,
                6.71887582,
                6.90949618,
                7.08351894,
                7.24360435,
                7.39182030,
                7.52980604,
                7.65888308,
                7.78013233,
                7.89444915,
                8.00258360,
                8.10517019,
                8.20275051,
            ]
        )

        self.known_yloglog = numpy.array(
            [
                33.11545196,
                74.50976691,
                132.46180783,
                206.97157474,
                298.03906763,
                405.66428649,
                529.84723134,
                670.58790216,
                827.88629897,
                1001.74242175,
                1192.15627051,
                1399.12784525,
                1622.65714598,
                1862.74417268,
                2119.38892536,
                2392.59140402,
                2682.35160865,
                2988.66953927,
                3311.54519587,
                3650.97857845,
            ]
        )

        self.known_ylinlog = numpy.array(
            [
                2.44691932e02,
                6.65141633e02,
                1.80804241e03,
                4.91476884e03,
                1.33597268e04,
                3.63155027e04,
                9.87157710e04,
                2.68337287e05,
                7.29416370e05,
                1.98275926e06,
                5.38969848e06,
                1.46507194e07,
                3.98247844e07,
                1.08254988e08,
                2.94267566e08,
                7.99902177e08,
                2.17435955e09,
                5.91052206e09,
                1.60664647e10,
                4.36731791e10,
            ]
        )

    def test_linlin(self):
        ylinlin = algo._estimate_from_fit(
            self.x, self.slope, self.intercept, xlog=False, ylog=False
        )
        nptest.assert_array_almost_equal(ylinlin, self.known_ylinlin)

    def test_loglin(self):
        yloglin = algo._estimate_from_fit(
            self.x, self.slope, self.intercept, xlog=True, ylog=False
        )
        nptest.assert_array_almost_equal(yloglin, self.known_yloglin)

    def test_loglog(self):
        yloglog = algo._estimate_from_fit(
            self.x, self.slope, self.intercept, xlog=True, ylog=True
        )
        nptest.assert_array_almost_equal(yloglog, self.known_yloglog)

    def test_linlog(self):
        ylinlog = algo._estimate_from_fit(
            self.x, self.slope, self.intercept, xlog=False, ylog=True
        )
        diff = numpy.abs(ylinlog - self.known_ylinlog) / self.known_ylinlog
        nptest.assert_array_almost_equal(diff, numpy.zeros(self.x.shape[0]), decimal=5)
