"""
scikitplot._xp_core_lib.validation - Utility Functions for Validation

This module provides utility functions designed to validate inputs and 
parameters within the scikit-plot library. These functions assist in ensuring 
that inputs conform to expected formats and constraints, and support various 
data handling tasks. The utilities in this module are essential for robust 
data validation and manipulation, enhancing the reliability and usability 
of the library.

Functions and classes provided include:
- Validation and type-checking utilities
- Functions for handling numpy arrays and NaN values
- Decorators for managing deprecated or positional arguments
- Utilities for inspecting function signatures

This module is part of the scikit-plot library and is intended for internal use 
to facilitate the validation and processing of inputs.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import warnings
import re
import math
import numbers
import inspect
import operator
import functools

from collections import namedtuple
from contextlib import contextmanager
from typing import (
  TYPE_CHECKING,
  TypeVar,
  Callable,
  Iterator,
  Optional,
  Tuple,
  Union,
  List,
)

import numpy as np
import numpy.ma as npma

import scipy.sparse

from scikitplot._xp_core_lib._array_api import array_namespace

__all__ = [
    '_lazywhere',
    '_lazyselect',
    '_prune_array',
    'float_factorial',
    'check_random_state',
    'rng_integers',
    '_fixed_default_rng',
    '_rng_html_rewrite',
    '_asarray_validated',
    '_validate_int',
    'FullArgSpec',
    'getfullargspec_no_self',
    '_FunctionWrapper',
    '_PythonFuncWrapper',
    '_python_func_wrapper',
    'MapWrapper',
    '_argmin',
    '_first_nonnan',
    '_nan_allsame',
    '_get_nan',
    '_rename_parameter',
]
_all_ignore = [
  'absolute_import', 'division', 'print_function', 'unicode_literals', 'mpl',
]


# Determine numpy exception classes
if np.lib.NumpyVersion(np.__version__) >= '1.25.0':
    from numpy.exceptions import (
        AxisError, ComplexWarning, VisibleDeprecationWarning,
        DTypePromotionError
    )
else:
    from numpy import (  # type: ignore[attr-defined, no-redef]
        AxisError, ComplexWarning, VisibleDeprecationWarning  # noqa: F401
    )
    DTypePromotionError = TypeError  # type: ignore


# Determine numpy integer types
if np.lib.NumpyVersion(np.__version__) >= "2.0.0.dev0":
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                r".*In the future `np\.long` will be defined as.*",
                FutureWarning,
            )
            np_long = np.long  # type: ignore[attr-defined]
            np_ulong = np.ulong  # type: ignore[attr-defined]
    except AttributeError:
        np_long = np.int_
        np_ulong = np.uint
else:
    np_long = np.int_
    np_ulong = np.uint

IntNumber = Union[int, np.integer]
DecimalNumber = Union[float, np.floating, np.integer]

# Determine whether to copy arrays
copy_if_needed: Optional[bool]

if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    copy_if_needed = None
elif np.lib.NumpyVersion(np.__version__) < "1.28.0":
    copy_if_needed = False
else:
    try:
        np.array([1]).__array__(copy=None)  # type: ignore[call-overload]
        copy_if_needed = None
    except TypeError:
        copy_if_needed = False


# Define random number generator types
if TYPE_CHECKING:
    SeedType = Union[IntNumber, np.random.Generator, np.random.RandomState, None]
    GeneratorType = TypeVar("GeneratorType", bound=Union[np.random.Generator, np.random.RandomState])


def _lazywhere(
  cond: np.ndarray,
  arrays: Tuple[np.ndarray],
  f: Callable,
  fillvalue: Optional[object] = None,
  f2: Optional[Callable] = None
) -> np.ndarray:
    """
    Return elements chosen from two possibilities depending on a condition.
    
    Equivalent to ``f(*arrays) if cond else fillvalue`` performed elementwise.

    Parameters
    ----------
    cond : np.ndarray
        The condition (expressed as a boolean array).
    arrays : tuple of np.ndarray
        Arguments to `f` (and `f2`). Must be broadcastable with `cond`.
    f : Callable
        Where `cond` is True, output will be ``f(arr1[cond], arr2[cond], ...)``.
    fillvalue : object, optional
        Value with which to fill output array where `cond` is not True.
    f2 : Callable, optional
        Output will be ``f2(arr1[cond], arr2[cond], ...)`` where `cond` is not True.

    Returns
    -------
    np.ndarray
        An array with elements from the output of `f` where `cond` is True
        and `fillvalue` (or elements from the output of `f2`) elsewhere.
    """
    xp = array_namespace(cond, *arrays)

    if (f2 is fillvalue is None) or (f2 is not None and fillvalue is not None):
        raise ValueError("Exactly one of `fillvalue` or `f2` must be given.")

    args = xp.broadcast_arrays(cond, *arrays)
    bool_dtype = xp.asarray([True]).dtype
    # cond, arrays = xp.astype(args[0], bool_dtype, copy=False), args[1:]
    cond, arrays = args[0].astype(bool_dtype, copy=False), args[1:]

    temp1 = xp.asarray(f(*(arr[cond] for arr in arrays)))

    if f2 is None:
        if isinstance(fillvalue, (bool, int, float, complex)):
            with np.errstate(invalid='ignore'):
                dtype = (temp1 * fillvalue).dtype
        else:
            dtype = xp.result_type(temp1.dtype, fillvalue)
        out = xp.full(cond.shape, dtype=dtype,
                      fill_value=xp.asarray(fillvalue, dtype=dtype))
    else:
        ncond = ~cond
        temp2 = xp.asarray(f2(*(arr[ncond] for arr in arrays)))
        dtype = xp.result_type(temp1, temp2)
        out = xp.empty(cond.shape, dtype=dtype)
        out[ncond] = temp2

    out[cond] = temp1

    return out


def _lazyselect(
  condlist: list,
  choicelist: list,
  arrays: Tuple[np.ndarray],
  default: object = 0
) -> np.ndarray:
    """
    Mimic `np.select(condlist, choicelist)`.

    Assumes that all `arrays` are of the same shape or can be broadcasted together.

    Parameters
    ----------
    condlist : list of np.ndarray
        List of boolean arrays to choose from.
    choicelist : list of Callable
        List of functions to apply to the arrays based on conditions.
    arrays : tuple of np.ndarray
        Arguments to functions in `choicelist`.
    default : object, optional
        Value to use if no conditions are met.

    Returns
    -------
    np.ndarray
        Array with values chosen based on conditions and functions.
    """
    arrays = np.broadcast_arrays(*arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=default, dtype=tcode)
    for func, cond in zip(choicelist, condlist):
        if np.all(cond is False):
            continue
        cond, _ = np.broadcast_arrays(cond, arrays[0])
        temp = tuple(np.extract(cond, arr) for arr in arrays)
        np.place(out, cond, func(*temp))
    return out


def _prune_array(
  array: np.ndarray
) -> np.ndarray:
    """
    Return an array equivalent to the input array. If the input
    array is a view of a much larger array, copy its contents to a
    newly allocated array. Otherwise, return the input unchanged.
    
    Parameters
    ----------
    array : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        Either the input array or a copy of its contents.
    """
    if array.base is not None and array.size < array.base.size // 2:
        return array.copy()
    return array


def float_factorial(n: int) -> float:
    """
    Compute the factorial and return as a float.

    Returns infinity when result is too large for a double.
    
    Parameters
    ----------
    n : int
        The input integer.

    Returns
    -------
    float
        The factorial of `n` as a float.
    """
    return float(math.factorial(n)) if n < 171 else np.inf


def check_random_state(
  seed: Optional[Union[int, np.random.Generator, np.random.RandomState]]
) -> Union[np.random.Generator, np.random.RandomState]:
    """
    Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``np.random.Generator`` or ``np.random.RandomState``
        instance then that instance is used.

    Returns
    -------
    Union[np.random.Generator, np.random.RandomState]
        Random number generator.

    Raises
    ------
    ValueError
        If `seed` is not valid.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed

    raise ValueError(f"'{seed}' cannot be used to seed a numpy.random.RandomState instance")


def rng_integers(
    gen: Union[np.random.RandomState, np.random.Generator],
    low: Union[int, np.ndarray],
    high: Optional[Union[int, np.ndarray]] = None,
    size: Optional[Union[int, tuple]] = None,
    dtype: str = 'int64',
    endpoint: bool = False
) -> Union[int, np.ndarray]:
    """
    Draw random integers from a specified range.

    This function generates random integers from a discrete uniform distribution 
    defined by the range `[low, high)` or `[low, high]` if `endpoint=True`. It
    serves as a replacement for `RandomState.randint` (with `endpoint=False`) 
    and `RandomState.random_integers` (with `endpoint=True`).

    Parameters
    ----------
    gen : Union[np.random.RandomState, np.random.Generator]
        Random number generator.
        If `None`, the default `np.random.RandomState` singleton is used.
    low : Union[int, np.ndarray]
        Lowest (signed) integer(s) to be drawn from the distribution.
        If `high` is `None`, this parameter is used for `high`.
    high : Optional[Union[int, np.ndarray]], default=None
        One above the largest (signed) integer to be drawn from the distribution.
        If `None`, results are from `0` to `low`.
    size : Optional[Union[int, tuple]], default=None
        Output shape. If the shape is `(m, n, k)`, then `m * n * k` samples are drawn.
        Defaults to `None`, in which case a single value is returned.
    dtype : str, default='int64'
        Desired dtype of the result. Default is `'int64'`.
        Dtypes are specified by name (e.g., `'int64'`, `'int'`).
    endpoint : bool, default=False
        If `True`, samples from the
        interval `[low, high]` (inclusive)
        instead of `[low, high)` (exclusive).

    Returns
    -------
    Union[int, np.ndarray]
        An array of random integers with the specified shape,
        or a single integer if `size` is not provided.
    """
    if isinstance(gen, np.random.Generator):
        return gen.integers(low, high=high, size=size, dtype=dtype, endpoint=endpoint)
    else:
        if gen is None:
            gen = np.random.mtrand._rand
        if endpoint:
            if high is None:
                return gen.randint(low + 1, size=size, dtype=dtype)
            return gen.randint(low, high=high + 1, size=size, dtype=dtype)
        return gen.randint(low, high=high, size=size, dtype=dtype)


@contextmanager
def _fixed_default_rng(
  seed: int = 1638083107694713882823079058616272161
) -> Iterator[None]:
    """
    Context manager to fix the seed of `np.random.default_rng`.

    This context manager temporarily overrides `np.random.default_rng` to use
    a fixed seed for reproducibility. The original function is restored after 
    the context exits.

    Parameters
    ----------
    seed : int, optional
        Seed value for the random number generator. Defaults to a large fixed integer.

    Yields
    ------
    None
    """
    orig_fun = np.random.default_rng
    np.random.default_rng = lambda seed=seed: orig_fun(seed)
    try:
        yield
    finally:
        np.random.default_rng = orig_fun


def _rng_html_rewrite(
  func: Callable[..., List[str]]
) -> Callable[..., List[str]]:
    """
    Decorator to modify the HTML rendering of `np.random.default_rng`.
    This is intended to decorate
    ``numpydoc.docscrape_sphinx.SphinxDocString._str_examples``.

    This decorator adjusts the HTML output of example code in Sphinx documentation
    by replacing specific instances of `np.random.default_rng` with a placeholder.
    This ensures consistent appearance in the documentation.

    Examples are only run by Sphinx when there are plot involved. Even so,
    it does not change the result values getting printed.

    Parameters
    ----------
    func : Callable[..., List[str]]
        Function to be wrapped. Its output will be modified to rewrite the HTML rendering.

    Returns
    -------
    Callable[..., List[str]]
        Wrapped function with modified HTML rendering for `np.random.default_rng`.
    """
    pattern = re.compile(r'np.random.default_rng\((0x[0-9A-F]+|\d+)\)', re.I)

    def _wrapped(*args, **kwargs) -> List[str]:
        res = func(*args, **kwargs)
        lines = [re.sub(pattern, 'np.random.default_rng()', line) for line in res]
        return lines

    return _wrapped


def _asarray_validated(
  a,
  check_finite=True,
  sparse_ok=False,
  objects_ok=False,
  mask_ok=False,
  as_inexact=False
) -> np.ndarray:
    """
    Helper function for validating and converting input arrays.
    
    Parameters
    ----------
    a : array_like
        The input array-like object.
    check_finite : bool, optional
        Whether to check that the input contains only finite numbers.
    sparse_ok : bool, optional
        If True, allows sparse matrices.
    objects_ok : bool, optional
        If True, allows arrays with dtype 'O'.
    mask_ok : bool, optional
        If True, allows masked arrays.
    as_inexact : bool, optional
        If True, converts the array to an inexact dtype (e.g., float64).
    
    Returns
    -------
    np.ndarray
        Validated and converted array.

    Raises
    ------
    ValueError
        If input contains non-finite numbers and `check_finite` is True.
    """
    if isinstance(a, npma.MaskedArray) and not mask_ok:
        raise TypeError("masked array not supported")

    if isinstance(a, scipy.sparse.spmatrix) and not sparse_ok:
        raise TypeError("sparse matrices not supported")

    a = np.asarray(a)

    if not objects_ok and a.dtype == np.dtype('O'):
        raise TypeError("object arrays are not supported")

    if as_inexact and not np.issubdtype(a.dtype, np.inexact):
        return a.astype(np.float64)

    if check_finite:
        if np.any(np.isnan(a)) or np.any(np.isinf(a)):
            raise ValueError("Input contains NaN or infinity")

    return a


def _validate_int(k: int, name: str, minimum: Optional[int] = None) -> int:
    """
    Validate a scalar integer.

    This function checks whether the input is a valid integer and optionally 
    whether it meets a minimum value requirement.

    Parameters
    ----------
    k : int
        The value to be validated.
    name : str
        The name of the parameter to be used in error messages.
    minimum : int, optional
        An optional lower bound for the value.

    Returns
    -------
    int
        The validated integer.

    Raises
    ------
    TypeError
        If `k` is not an integer.
    ValueError
        If `k` is less than `minimum`, when `minimum` is provided.
    """
    try:
        k = operator.index(k)
    except TypeError:
        raise TypeError(f'{name} must be an integer.') from None
    if minimum is not None and k < minimum:
        raise ValueError(f'{name} must be an integer not less than {minimum}') from None
    return k


# Add a replacement for inspect.getfullargspec()/
# The version below is borrowed from Django,
# https://github.com/django/django/pull/4846.

# Note an inconsistency between inspect.getfullargspec(func) and
# inspect.signature(func). If `func` is a bound method, the latter does *not*
# list `self` as a first argument, while the former *does*.
# Hence, cook up a common ground replacement: `getfullargspec_no_self` which
# mimics `inspect.getfullargspec` but does not list `self`.
#
# This way, the caller code does not need to know whether it uses a legacy
# .getfullargspec or a bright and shiny .signature.

FullArgSpec = namedtuple('FullArgSpec',
                         ['args', 'varargs', 'varkw', 'defaults',
                          'kwonlyargs', 'kwonlydefaults', 'annotations'])


def getfullargspec_no_self(func: Callable) -> FullArgSpec:
    """
    Replacement for `inspect.getfullargspec` that omits the 'self' parameter 
    if `func` is a bound method.

    Parameters
    ----------
    func : Callable
        A callable to inspect.

    Returns
    -------
    FullArgSpec
        A named tuple containing argument specification of the function.

    Notes
    -----
    if the first argument of `func` is self, it is *not*, I repeat *not*,
    included in fullargspec.args.
    This is done for consistency between inspect.getargspec() under
    Python 2.x, and inspect.signature() under Python 3.x.

    """
    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY]
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = tuple(
        p.default for p in sig.parameters.values()
        if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
            p.default is not p.empty)
    ) or None
    kwonlyargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    kwdefaults = {p.name: p.default for p in sig.parameters.values()
                  if p.kind == inspect.Parameter.KEYWORD_ONLY and
                  p.default is not p.empty}
    annotations = {p.name: p.annotation for p in sig.parameters.values()
                   if p.annotation is not p.empty}
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                       kwdefaults or None, annotations)



class _FunctionWrapper:
    """
    Wraps a function to enable pickling.

    This class allows a function to be stored and called later with the same 
    arguments, while ensuring it can be pickled.
    """
    def __init__(self, f: Callable, args: Optional[list] = None):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        return self.f(x, *self.args)


class _PythonFuncWrapper:
    """
    Wraps a function to enable pickling.

    This class allows a function to be stored and called later with the same
    arguments, while ensuring it can be pickled.
    """
    def __init__(self, f: Callable, args: Optional[list] = None):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, *args, **kwargs):
        return self.f(*args, *self.args, **kwargs)
    
    def __reduce__(self):
        # This is required for pickling to work properly
        return (self.__class__, (self.f, self.args))


def _python_func_wrapper(f: Callable):
    """
    A decorator that wraps a function in a _PythonFuncWrapper to make it picklable.

    Parameters
    ----------
    f : Callable
        The function to be wrapped.

    Returns
    -------
    Callable
        The wrapped function that can be pickled.
    """
    def wrapper(*args, **kwargs):
        return _PythonFuncWrapper(f)(*args, **kwargs)
    return wrapper


class MapWrapper:
    """
    Parallelisation wrapper for working with map-like callables, such as
    `multiprocessing.Pool.map`.

    Parameters
    ----------
    pool : int or map-like callable
        If `pool` is an integer, then it specifies the number of threads to
        use for parallelization. If ``int(pool) == 1``, then no parallel
        processing is used and the map builtin is used.
        If ``pool == -1``, then the pool will utilize all available CPUs.
        If `pool` is a map-like callable that follows the same
        calling sequence as the built-in map function, then this callable is
        used for parallelization.
    """
    def __init__(self, pool=1):
        self.pool = None
        self._mapfunc = map
        self._own_pool = False

        if callable(pool):
            self.pool = pool
            self._mapfunc = self.pool
        else:
            from multiprocessing import Pool
            # user supplies a number
            if int(pool) == -1:
                # use as many processors as possible
                self.pool = Pool()
                self._mapfunc = self.pool.map
                self._own_pool = True
            elif int(pool) == 1:
                pass
            elif int(pool) > 1:
                # use the number of processors requested
                self.pool = Pool(processes=int(pool))
                self._mapfunc = self.pool.map
                self._own_pool = True
            else:
                raise RuntimeError("Number of workers specified must be -1,"
                                   " an int >= 1, or an object with a 'map' "
                                   "method")

    def __enter__(self):
        return self

    def terminate(self):
        if self._own_pool:
            self.pool.terminate()

    def join(self):
        if self._own_pool:
            self.pool.join()

    def close(self):
        if self._own_pool:
            self.pool.close()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._own_pool:
            self.pool.close()
            self.pool.terminate()

    def __call__(self, func, iterable):
        # only accept one iterable because that's all Pool.map accepts
        try:
            return self._mapfunc(func, iterable)
        except TypeError as e:
            # wrong number of arguments
            raise TypeError("The map-like callable must be of the"
                            " form f(func, iterable)") from e


def _argmin(
  a: np.ndarray,
  keepdims: bool = False,
  axis: Optional[int] = None
) -> np.ndarray:
    """
    Compute the index of the minimum value along an axis with optional `keepdims` parameter.

    Parameters
    ----------
    a : np.ndarray
        The input array.
    keepdims : bool, optional
        Whether to retain reduced dimensions.
    axis : int, optional
        Axis or axes along which to operate. If None, operate on the flattened array.

    Returns
    -------
    np.ndarray
        Indices of the minimum values along the specified axis.

    References
    ----------
    See https://github.com/numpy/numpy/issues/8710
    If axis is not None, a.shape[axis] must be greater than 0.
    """
    res = np.argmin(a, axis=axis)
    if keepdims and axis is not None:
        res = np.expand_dims(res, axis=axis)
    return res


def _first_nonnan(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Return the first non-NaN value along the specified axis.
    If a slice is all nan, nan is returned for that slice.
    The shape of the return value corresponds to ``keepdims=True``.

    Parameters
    ----------
    a : np.ndarray
        The input array.
    axis : int
        The axis along which to search for the first non-NaN value.

    Returns
    -------
    np.ndarray
        An array with the first non-NaN values along the specified axis.

    Examples
    --------
    >>> import numpy as np
    >>> nan = np.nan
    >>> a = np.array([[ 3.,  3., nan,  3.],
                      [ 1., nan,  2.,  4.],
                      [nan, nan,  9., -1.],
                      [nan,  5.,  4.,  3.],
                      [ 2.,  2.,  2.,  2.],
                      [nan, nan, nan, nan]])
    >>> _first_nonnan(a, axis=0)
    array([[3., 3., 2., 3.]])
    >>> _first_nonnan(a, axis=1)
    array([[ 3.],
           [ 1.],
           [ 9.],
           [ 5.],
           [ 2.],
           [nan]])
    """
    k = _argmin(np.isnan(a), axis=axis, keepdims=True)
    return np.take_along_axis(a, k, axis=axis)


def _nan_allsame(
  a: np.ndarray,
  axis: Optional[int] = None,
  keepdims: bool = False
) -> np.ndarray:
    """
    Determine if the values along an axis are all the same.
    nan values are ignored.
    `a` must be a numpy array.
    `axis` is assumed to be normalized; that is, 0 <= axis < a.ndim.

    For an axis of length 0, the result is True.  That is, we adopt the
    convention that ``allsame([])`` is True. (There are no values in the
    input that are different.)

    `True` is returned for slices that are all nan--not because all the
    values are the same, but because this is equivalent to ``allsame([])``.

    Parameters
    ----------
    a : np.ndarray
        The input array.
    axis : int, optional
        The axis or axes along which to check for equality. If None, flattened array is used.
    keepdims : bool, optional
        Whether to retain reduced dimensions.

    Returns
    -------
    np.ndarray
        A boolean array indicating if all values along the specified axis are the same.

    Examples
    --------
    >>> from numpy import nan, array
    >>> a = array([[ 3.,  3., nan,  3.],
    ...            [ 1., nan,  2.,  4.],
    ...            [nan, nan,  9., -1.],
    ...            [nan,  5.,  4.,  3.],
    ...            [ 2.,  2.,  2.,  2.],
    ...            [nan, nan, nan, nan]])
    >>> _nan_allsame(a, axis=1, keepdims=True)
    array([[ True],
           [False],
           [False],
           [False],
           [ True],
           [ True]])
    """
    if axis is None:
        if a.size == 0:
            return True
        a = a.ravel()
        axis = 0
    else:
        shp = a.shape
        if shp[axis] == 0:
            shp = shp[:axis] + (1,)*keepdims + shp[axis + 1:]
            return np.full(shp, fill_value=True, dtype=bool)
    a0 = _first_nonnan(a, axis=axis)
    return ((a0 == a) | np.isnan(a)).all(axis=axis, keepdims=keepdims)


def _get_nan(*data, xp=None) -> np.ndarray:
    """
    Return a NaN value of the appropriate dtype based on the provided data.

    Parameters
    ----------
    data : array_like
        Data from which to infer the appropriate dtype.
    xp : Optional[Type], optional
        The module or namespace to use for array operations. If None, defaults to numpy.

    Returns
    -------
    np.ndarray
        A NaN value of the inferred dtype.
    """
    xp = array_namespace(*data) if xp is None else xp
    # Get NaN of appropriate dtype for data
    data = [xp.asarray(item) for item in data]
    try:
        min_float = getattr(xp, 'float16', xp.float32)
        dtype = xp.result_type(*data, min_float)  # must be at least a float
    except DTypePromotionError:
        # fallback to float64
        dtype = xp.float64
    return xp.asarray(xp.nan, dtype=dtype)[()]


def _rename_parameter(
  old_name: str,
  new_name: str,
  dep_version: Optional[str] = None
) -> Callable:
    """
    Generate Decorator for maintaining backward compatibility with renamed parameters.

    Apply the decorator generated by `_rename_parameter` to functions with a
    recently renamed parameter to maintain backward-compatibility.

    After decoration, the function behaves as follows:
    If only the new parameter is passed into the function, behave as usual.
    If only the old parameter is passed into the function (as a keyword), raise
    a DeprecationWarning if `dep_version` is provided, and behave as usual
    otherwise.
    If both old and new parameters are passed into the function, raise a
    DeprecationWarning if `dep_version` is provided, and raise the appropriate
    TypeError (function got multiple values for argument).

    Parameters
    ----------
    old_name : str
        The old name of the parameter.
    new_name : str
        The new name of the parameter.
    dep_version : str, optional
        Version of Scikitplot in which old parameter was deprecated in the format
        'X.Y.Z'. If supplied, the deprecation message will indicate that
        support for the old parameter will be removed in version 'X.Y+2.Z'

    Returns
    -------
    Callable
        The decorated function with backward-compatible parameter handling.

    Notes
    -----
    Untested with functions that accept *args. Probably won't work as written.

    """
    def decorator(fun: Callable) -> Callable:
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                if dep_version:
                    end_version = dep_version.split('.')
                    end_version[1] = str(int(end_version[1]) + 2)
                    end_version = '.'.join(end_version)
                    message = (f"Use of keyword argument `{old_name}` is "
                               f"deprecated and replaced by `{new_name}`.  "
                               f"Support for `{old_name}` will be removed "
                               f"in version {end_version}.")
                    warnings.warn(message, DeprecationWarning, stacklevel=2)
                if new_name in kwargs:
                    message = (f"{fun.__name__}() got multiple values for "
                               f"argument now known as `{new_name}`")
                    raise TypeError(message)
                kwargs[new_name] = kwargs.pop(old_name)
            return fun(*args, **kwargs)
        return wrapper
    return decorator