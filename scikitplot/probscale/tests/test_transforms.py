import numpy

import pytest
import numpy.testing as nptest

from ..probscale import _minimal_norm
from .. import transforms


def test__mask_out_of_bounds():
    known = numpy.array([numpy.nan, numpy.nan, 0.1, 0.5, 0.9, numpy.nan, numpy.nan])
    x = [-0.1, 0, 0.1, 0.5, 0.9, 1.0, 1.1]
    result = transforms._mask_out_of_bounds(x)
    nptest.assert_array_equal(known, result)


def test__clip_out_of_bounds():
    known = numpy.array([0.0, 0.0, 0.1, 0.5, 0.9, 1.0, 1.0])
    x = [-0.1, 0, 0.1, 0.5, 0.9, 1.0, 1.1]
    result = transforms._clip_out_of_bounds(x)
    diff = numpy.abs(result - known)
    assert numpy.all(diff < 0.0001)


@pytest.mark.parametrize(
    "trans",
    [
        transforms.ProbTransform(_minimal_norm),
        transforms.QuantileTransform(_minimal_norm),
    ],
)
def test_transform_input_dims(trans):
    assert trans.input_dims == 1


@pytest.mark.parametrize(
    "trans",
    [
        transforms.ProbTransform(_minimal_norm),
        transforms.QuantileTransform(_minimal_norm),
    ],
)
def test_transform_output_dims(trans):
    assert trans.output_dims == 1


@pytest.mark.parametrize(
    "trans",
    [
        transforms.ProbTransform(_minimal_norm),
        transforms.QuantileTransform(_minimal_norm),
    ],
)
def test_transform_is_separable(trans):
    assert trans.is_separable


@pytest.mark.parametrize(
    "trans",
    [
        transforms.ProbTransform(_minimal_norm),
        transforms.QuantileTransform(_minimal_norm),
    ],
)
def test_transform_has_inverse(trans):
    assert trans.has_inverse


@pytest.mark.parametrize(
    "trans",
    [
        transforms.ProbTransform(_minimal_norm),
        transforms.QuantileTransform(_minimal_norm),
    ],
)
def test_transform_dist(trans):
    trans.dist == _minimal_norm


@pytest.mark.parametrize(
    ("trans", "known_trans_na"),
    [
        (transforms.ProbTransform(_minimal_norm), -2.569150498),
        (transforms.QuantileTransform(_minimal_norm), 69.1464492),
    ],
)
def test_transform_non_affine(trans, known_trans_na):
    diff = numpy.abs(trans.transform_non_affine([0.5]) - known_trans_na)
    assert numpy.all(diff < 0.0001)


@pytest.mark.parametrize(
    ("trans", "inver_cls"),
    [
        (transforms.ProbTransform(_minimal_norm), transforms.QuantileTransform),
        (transforms.QuantileTransform(_minimal_norm), transforms.ProbTransform),
    ],
)
def test_transform_inverted(trans, inver_cls):
    t_inv = trans.inverted()
    assert isinstance(t_inv, inver_cls)
    assert trans.dist == t_inv.dist
    assert trans.as_pct == t_inv.as_pct
    assert trans.out_of_bounds == t_inv.out_of_bounds


@pytest.mark.parametrize(
    "cls", [transforms.ProbTransform, transforms.QuantileTransform]
)
def test_bad_out_of_bounds(cls):
    with pytest.raises(ValueError):
        cls(_minimal_norm, out_of_bounds="junk")


@pytest.mark.parametrize(
    "cls", [transforms.ProbTransform, transforms.QuantileTransform]
)
@pytest.mark.parametrize(
    ("method", "func"),
    [
        ("clip", transforms._clip_out_of_bounds),
        ("mask", transforms._mask_out_of_bounds),
        ("junk", None),
    ],
)
def test_out_of_bounds(cls, method, func):
    if func is None:
        with pytest.raises(ValueError):
            cls(_minimal_norm, out_of_bounds=method)
    else:
        t = cls(_minimal_norm, out_of_bounds=method)
        assert t.out_of_bounds == method
        assert t._handle_out_of_bounds == func
