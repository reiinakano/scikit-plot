import numpy

import pytest

from .. import formatters


@pytest.mark.parametrize("fmtr", [formatters.PctFormatter, formatters.ProbFormatter])
def test_base_class_of_formatter(fmtr):
    assert issubclass(fmtr, formatters._FormatterMixin)


@pytest.mark.parametrize(
    ("pct", "expected"),
    [
        (0.0301, "0.03"),
        (0.20, "0.2"),
        (0.100, "0.1"),
        (10.000, "10"),
        (5.00, "5"),
        (50.00, "50"),
        (99.000, "99"),
        (99.1, "99.1"),
        (99.99, "99.99"),
    ],
)
def test__call___PctFormatter(pct, expected):
    fmt = formatters.PctFormatter()
    assert fmt(pct) == expected


@pytest.mark.parametrize(
    ("prob", "expected"),
    [
        (0.000301, "0.0003"),
        (0.001000, "0.001"),
        (0.100000, "0.10"),
        (0.050000, "0.05"),
        (0.500000, "0.50"),
        (0.990000, "0.99"),
        (0.991000, "0.991"),
        (0.999900, "0.9999"),
    ],
)
def test__call___ProbFormmater(prob, expected):
    fmt = formatters.ProbFormatter()
    assert fmt(prob) == expected


@pytest.mark.parametrize(
    ("value", "N", "expected", "forceint"),
    [
        (1234.56, 3, "1,230", False),
        (1234.56, 4, "1,235", False),
        ("1.23", 3, "1.23", False),
        (numpy.nan, 3, "NA", False),
        (numpy.inf, 3, "NA", False),
        (0, 3, "0", False),
        (1234.56, 8, "1,234.5600", False),
        (1.23456e8, 3, "1.23e+08", False),
        (1234.56, 3, "1,235", True),
        (0.123456, 3, "0.123", False),
        (0.123456, 4, "0.1235", False),
        ("0.123456", 3, "0.123", False),
        (numpy.nan, 3, "NA", False),
        (numpy.inf, 3, "NA", False),
        (0, 3, "0", False),
        (0.123456, 8, "0.12345600", False),
        (1.23456e-7, 3, "1.23e-07", False),
        (0.123456, 3, "0", True),
    ],
)
def test__sig_figs(value, N, expected, forceint):
    fmt = formatters._FormatterMixin()
    assert fmt._sig_figs(value, N, forceint=forceint) == expected


@pytest.mark.parametrize("N", [-1, 0, 0.5])
def test__sig_figs_errors(N):
    fmt = formatters._FormatterMixin()
    with pytest.raises(ValueError):
        fmt._sig_figs(1.23, N)
