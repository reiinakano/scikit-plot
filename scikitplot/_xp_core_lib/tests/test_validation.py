import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

from hypothesis import reproduce_failure  # noqa: F401

import re
import math
import fractions
# import contextlib
import multiprocessing

from scikitplot.conftest import array_api_compatible, skip_xp_invalid_arg
skip_xp_backends = pytest.mark.skip_xp_backends

from .._array_api import (
  SKPLT_ARRAY_API,
  SKPLT_DEVICE,
  xp_assert_equal,
  xp_assert_close,
  is_array_api_strict,
)
from ..validation import (
  _lazywhere,
  check_random_state,
  _validate_int,
  MapWrapper,
  FullArgSpec,
  getfullargspec_no_self,
  rng_integers,
  _rename_parameter,
)


class TestLazywhere:
    n_arrays = hypothesis.strategies.integers(min_value=1, max_value=3)
    rng_seed = hypothesis.strategies.integers(min_value=1000000000, max_value=9999999999)
    dtype = hypothesis.strategies.sampled_from((np.float32, np.float64))
    p = hypothesis.strategies.floats(min_value=0, max_value=1)
    data = hypothesis.strategies.data()

    @pytest.mark.fail_slow(10)
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')  # overflows, etc.
    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible  # xp
    @hypothesis.settings(
        suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
        deadline=600,  # Set a higher deadline from 550ms, default 200
    )
    @hypothesis.given(
        n_arrays=n_arrays,
        rng_seed=rng_seed,
        dtype=dtype,
        p=p,
        data=data
    )
    def test_basic(self, n_arrays, rng_seed, dtype, p, data, xp):
        mbs = npst.mutually_broadcastable_shapes(
          num_shapes=n_arrays+1,
          min_side=0
        )
        input_shapes, result_shape = data.draw(mbs)
        cond_shape, *shapes = input_shapes
        elements = {'allow_subnormal': False}  # cupy/cupy#8382
        fillvalue = xp.asarray(data.draw(npst.arrays(
            dtype=dtype,
            shape=tuple(),
            elements=elements,
        )))
        float_fillvalue = float(fillvalue)
        arrays = [
            xp.asarray(data.draw(npst.arrays(
                dtype=dtype,
                shape=shape,
            )))
            for shape in shapes
        ]

        def f(*args):
            return sum(arg for arg in args)

        def f2(*args):
            return sum(arg for arg in args) / 2

        rng = np.random.default_rng(rng_seed)
        cond = xp.asarray(rng.random(size=cond_shape) > p)

        res1 = _lazywhere(cond, arrays, f, fillvalue).astype(dtype)
        res2 = _lazywhere(cond, arrays, f, f2=f2).astype(dtype)

        ref1 = xp.where(cond, f(*arrays), fillvalue).astype(dtype)
        ref2 = xp.where(cond, f(*arrays), f2(*arrays)).astype(dtype)

        # Ensure arrays are at least 1d to follow sane type promotion rules.
        # This can be removed when minimum supported NumPy is 2.0
        if xp == np:
            cond, fillvalue, *arrays = np.atleast_1d(cond, fillvalue, *arrays)

        if xp == np:  # because we ensured arrays are at least 1d
            ref1 = ref1.reshape(result_shape)
            ref2 = ref2.reshape(result_shape)
            # ref3 = ref3.reshape(result_shape)

        # Adjust tolerance for float32
        rtol = 1e-5 if dtype == np.float32 else 2e-16
      
        # Assert results with adjusted tolerance
        xp_assert_close(res1, ref1, rtol=rtol)
        xp_assert_equal(res2, ref2)
      
        if not is_array_api_strict(xp):
            res3 = _lazywhere(cond, arrays, f, float_fillvalue)
      
            # Array API standard doesn't currently define behavior when fillvalue is a
            # Python scalar. When it does, test can be run with array_api_strict, too.
            ref3 = xp.where(cond, f(*arrays), float_fillvalue)
      
            xp_assert_equal(res3, ref3)


class TestRandomState:
    def test_check_random_state(self):
        # If seed is None, return the RandomState singleton used by np.random.
        # If seed is an int, return a new RandomState instance seeded with seed.
        # If seed is already a RandomState instance, return it.
        # Otherwise raise ValueError.
        rsi = check_random_state(1)
        np_testing.assert_equal(type(rsi), np.random.RandomState)
        rsi = check_random_state(rsi)
        np_testing.assert_equal(type(rsi), np.random.RandomState)
        rsi = check_random_state(None)
        np_testing.assert_equal(type(rsi), np.random.RandomState)
        pytest.raises(ValueError, check_random_state, 'a')
        rg = np.random.Generator(np.random.PCG64())
        rsi = check_random_state(rg)
        np_testing.assert_equal(type(rsi), np.random.Generator)
    
    
    def test_rng_integers(self):
        rng = np.random.RandomState()
    
        # test that numbers are inclusive of high point
        arr = rng_integers(rng, low=2, high=5, size=100, endpoint=True)
        assert np.max(arr) == 5
        assert np.min(arr) == 2
        assert arr.shape == (100, )
    
        # test that numbers are inclusive of high point
        arr = rng_integers(rng, low=5, size=100, endpoint=True)
        assert np.max(arr) == 5
        assert np.min(arr) == 0
        assert arr.shape == (100, )
    
        # test that numbers are exclusive of high point
        arr = rng_integers(rng, low=2, high=5, size=100, endpoint=False)
        assert np.max(arr) == 4
        assert np.min(arr) == 2
        assert arr.shape == (100, )
    
        # test that numbers are exclusive of high point
        arr = rng_integers(rng, low=5, size=100, endpoint=False)
        assert np.max(arr) == 4
        assert np.min(arr) == 0
        assert arr.shape == (100, )
    
        # now try with np.random.Generator
        try:
            rng = np.random.default_rng()
        except AttributeError:
            return
    
        # test that numbers are inclusive of high point
        arr = rng_integers(rng, low=2, high=5, size=100, endpoint=True)
        assert np.max(arr) == 5
        assert np.min(arr) == 2
        assert arr.shape == (100, )
    
        # test that numbers are inclusive of high point
        arr = rng_integers(rng, low=5, size=100, endpoint=True)
        assert np.max(arr) == 5
        assert np.min(arr) == 0
        assert arr.shape == (100, )
    
        # test that numbers are exclusive of high point
        arr = rng_integers(rng, low=2, high=5, size=100, endpoint=False)
        assert np.max(arr) == 4
        assert np.min(arr) == 2
        assert arr.shape == (100, )
    
        # test that numbers are exclusive of high point
        arr = rng_integers(rng, low=5, size=100, endpoint=False)
        assert np.max(arr) == 4
        assert np.min(arr) == 0
        assert arr.shape == (100, )


class TestValidateInt:
    @pytest.mark.parametrize('n', [4, np.uint8(4), np.int16(4), np.array(4)])
    def test_validate_int(self, n):
        n = _validate_int(n, 'n')
        assert n == 4

    @pytest.mark.parametrize('n', [4.0, np.array([4]), fractions.Fraction(4, 1)])
    def test_validate_int_bad(self, n):
        with pytest.raises(TypeError, match='n must be an integer'):
            _validate_int(n, 'n')

    def test_validate_int_below_min(self):
        with pytest.raises(ValueError, match='n must be an integer not '
                                             'less than 0'):
            _validate_int(-1, 'n', 0)


class TestMapWrapper:
    def test_mapwrapper_serial(self):
        in_arg = np.arange(10.)
        out_arg = np.sin(in_arg)
    
        p = MapWrapper(1)
        np_testing.assert_(p._mapfunc is map)
        np_testing.assert_(p.pool is None)
        np_testing.assert_(p._own_pool is False)
        out = list(p(np.sin, in_arg))
        np_testing.assert_equal(out, out_arg)
    
        with pytest.raises(RuntimeError):
            p = MapWrapper(0)
    
    def test_pool(self):
        with multiprocessing.Pool(2) as p:
            p.map(math.sin, [1, 2, 3, 4])

    def test_mapwrapper_parallel(self):
        in_arg = np.arange(10.)
        out_arg = np.sin(in_arg)
    
        with MapWrapper(2) as p:
            out = p(np.sin, in_arg)
            np_testing.assert_equal(list(out), out_arg)
    
            np_testing.assert_(p._own_pool is True)
            np_testing.assert_(isinstance(p.pool, multiprocessing.pool.Pool))
            np_testing.assert_(p._mapfunc is not None)
    
        # the context manager should've closed the internal pool
        # check that it has by asking it to calculate again.
        with pytest.raises(Exception) as excinfo:
            p(np.sin, in_arg)
    
        np_testing.assert_(excinfo.type is ValueError)
    
        # can also set a PoolWrapper up with a map-like callable instance
        with multiprocessing.Pool(2) as p:
            q = MapWrapper(p.map)
    
            np_testing.assert_(q._own_pool is False)
            q.close()
    
            # closing the PoolWrapper shouldn't close the internal pool
            # because it didn't create it
            out = p.map(np.sin, in_arg)
            np_testing.assert_equal(list(out), out_arg)

    def test_getfullargspec_no_self(self):
        p = MapWrapper(1)
        argspec = getfullargspec_no_self(p.__init__)
        np_testing.assert_equal(argspec, FullArgSpec(['pool'], None, None, (1,), [],
                                          None, {}))
        argspec = getfullargspec_no_self(p.__call__)
        np_testing.assert_equal(argspec, FullArgSpec(['func', 'iterable'], None, None, None,
                                          [], None, {}))
    
        class _rv_generic:
            def _rvs(self, a, b=2, c=3, *args, size=None, **kwargs):
                return None
    
        rv_obj = _rv_generic()
        argspec = getfullargspec_no_self(rv_obj._rvs)
        np_testing.assert_equal(argspec, FullArgSpec(['a', 'b', 'c'], 'args', 'kwargs',
                                          (2, 3), ['size'], {'size': None}, {}))


class TestRenameParameter:
    # check that wrapper `_rename_parameter` for backward-compatible
    # keyword renaming works correctly

    # Example method/function that still accepts keyword `old`
    @_rename_parameter("old", "new")
    def old_keyword_still_accepted(self, new):
        return new

    # Example method/function for which keyword `old` is deprecated
    @_rename_parameter("old", "new", dep_version="1.9.0")
    def old_keyword_deprecated(self, new):
        return new

    def test_old_keyword_still_accepted(self):
        # positional argument and both keyword work identically
        res1 = self.old_keyword_still_accepted(10)
        res2 = self.old_keyword_still_accepted(new=10)
        res3 = self.old_keyword_still_accepted(old=10)
        assert res1 == res2 == res3 == 10

        # unexpected keyword raises an error
        message = re.escape("old_keyword_still_accepted() got an unexpected")
        with pytest.raises(TypeError, match=message):
            self.old_keyword_still_accepted(unexpected=10)

        # multiple values for the same parameter raises an error
        message = re.escape("old_keyword_still_accepted() got multiple")
        with pytest.raises(TypeError, match=message):
            self.old_keyword_still_accepted(10, new=10)
        with pytest.raises(TypeError, match=message):
            self.old_keyword_still_accepted(10, old=10)
        with pytest.raises(TypeError, match=message):
            self.old_keyword_still_accepted(new=10, old=10)

    def test_old_keyword_deprecated(self):
        # positional argument and both keyword work identically,
        # but use of old keyword results in DeprecationWarning
        dep_msg = "Use of keyword argument `old` is deprecated"
        res1 = self.old_keyword_deprecated(10)
        res2 = self.old_keyword_deprecated(new=10)
        with pytest.warns(DeprecationWarning, match=dep_msg):
            res3 = self.old_keyword_deprecated(old=10)
        assert res1 == res2 == res3 == 10

        # unexpected keyword raises an error
        message = re.escape("old_keyword_deprecated() got an unexpected")
        with pytest.raises(TypeError, match=message):
            self.old_keyword_deprecated(unexpected=10)

        # multiple values for the same parameter raises an error and,
        # if old keyword is used, results in DeprecationWarning
        message = re.escape("old_keyword_deprecated() got multiple")
        with pytest.raises(TypeError, match=message):
            self.old_keyword_deprecated(10, new=10)
        with pytest.raises(TypeError, match=message), \
                pytest.warns(DeprecationWarning, match=dep_msg):
            self.old_keyword_deprecated(10, old=10)
        with pytest.raises(TypeError, match=message), \
                pytest.warns(DeprecationWarning, match=dep_msg):
            self.old_keyword_deprecated(new=10, old=10)