from matplotlib import pyplot

import pytest

from .. import validate
from .. import algo


def test_axes_object_invalid():
    with pytest.raises(ValueError):
        validate.axes_object("junk")


def test_axes_object_with_ax():
    fig, ax = pyplot.subplots()
    fig1, ax1 = validate.axes_object(ax)
    assert isinstance(ax1, pyplot.Axes)
    assert isinstance(fig1, pyplot.Figure)
    assert ax1 is ax
    assert fig1 is fig


def test_axes_object_with_None():
    fig1, ax1 = validate.axes_object(None)
    assert isinstance(ax1, pyplot.Axes)
    assert isinstance(fig1, pyplot.Figure)


@pytest.mark.parametrize(
    ("which", "kwarg"),
    [
        ("x", "fitprobs"),
        ("y", "fitprobs"),
        ("y", "fitlogs"),
        ("both", "fitprobs"),
        ("both", "fitlogs"),
        (None, "fitprobs"),
        (None, "fitlogs"),
    ],
)
def test_fit_arguments_valid(which, kwarg):
    result = validate.fit_argument(which, kwarg)
    assert result == which


@pytest.mark.parametrize("kwarg", ["fitprobs", "fitlogs"])
def test_fit_arguments_invalid(kwarg):
    with pytest.raises(ValueError):
        validate.fit_argument("junk", kwarg)


@pytest.mark.parametrize(
    ("value", "error"), [("x", None), ("y", None), ("junk", ValueError)]
)
def test_axis_name(value, error):
    if error is not None:
        with pytest.raises(error):
            validate.axis_name(value, "axname")

    else:
        assert value == validate.axis_name(value, "axname")


@pytest.mark.parametrize(
    ("value", "expected", "error"),
    [
        ("PP", "pp", None),
        ("Qq", "qq", None),
        ("ProB", "prob", None),
        ("junk", None, ValueError),
    ],
)
def test_axis_type(value, expected, error):
    if error is not None:
        with pytest.raises(error):
            validate.axis_type(value)

    else:
        assert expected == validate.axis_type(value)


@pytest.mark.parametrize(
    ("value", "expected"), [(None, dict()), (dict(a=1, b="test"), dict(a=1, b="test"))]
)
def test_other_options(value, expected):
    assert validate.other_options(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(None, ""), ("test", "test")])
def test_axis_label(value, expected):
    result = validate.axis_label(value)
    assert result == expected


@pytest.mark.parametrize(
    ("value", "expected", "error"),
    [
        ("fit", algo._bs_fit, None),
        ("resids", None, NotImplementedError),
        ("junk", None, ValueError),
    ],
)
def test_estimator(value, expected, error):
    if error is not None:
        with pytest.raises(error):
            validate.estimator(value)
    else:
        est = validate.estimator(value)
        assert est is expected
