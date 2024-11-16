"""
.. highlight:: cython

Experimental API Functions by Cython
=====================================

This module provides Cython implementations of several mathematical functions often used in statistical and machine learning contexts, such as the expit (sigmoid) function, its logarithm, and the logit function. These functions are scalar and typed versions of functions commonly found in libraries like `scipy.special`.

The module leverages Cython's fused types to handle different numeric types (`double`, `float`, `long double`) in a single function definition, making the code both efficient and flexible.

Key Conventions:
----------------
- Functions that return multiple values in Python will return their outputs via pointers in the final arguments.
- Functions returning a single value in Python will return that value directly.

Error Handling:
---------------
- Functions return `nan` to indicate an error, but unlike their `scipy.special` counterparts, they do not emit warnings.

Usage:
------
The module is usable from Cython via::

    cimport scikitplot._experimental._cy_experimental

Available Functions:
--------------------
- :py:func:`~scikitplot.experimental.expit`:
  
  Computes the sigmoid of the input value.

        double expit(double)
        float expit(float)
        long double expit(long double)

- :py:func:`~scikitplot.experimental.log_expit`:
  
  Computes the logarithm of the sigmoid of the input value.

        double log_expit(double)
        float log_expit(float)
        long double log_expit(long double)

- :py:func:`~scikitplot._experimental.logit`:
  
  Computes the logit (inverse of the sigmoid) of the input value.

        double logit(double)
        float logit(float)
        long double logit(long double)
"""

cimport cython
from cython cimport nogil

from libc.math cimport (
    NAN,
    exp,
    log
)
from numpy cimport (
    npy_float,
    npy_double,
    npy_longdouble,
    npy_cdouble,
    npy_int, 
    npy_long
)

# Implement functions

cpdef dfg_number_t expit(dfg_number_t x0) noexcept nogil:
    """
    Compute the expit (sigmoid) function of the input value `x0`.

    The expit function is defined as::
    
        expit(x) = 1 / (1 + exp(-x))
    
    This function is widely used in logistic regression models and other
    areas of machine learning and statistics.

    .. versionadded:: 0.3.9

    Parameters:
    -----------
    x0 : dfg_number_t
        The input value, which can be a `double`, `float`, or `long double`.

    Returns:
    --------
    dfg_number_t
        The sigmoid of the input value, of the same type as `x0`.

    Example:
    --------
    >>> expit(0.5)
    0.6224593312018546
    """
    return 1.0 / (1.0 + exp(-x0))


cpdef dfg_number_t log_expit(dfg_number_t x0) noexcept nogil:
    """
    Compute the logarithm of the expit (sigmoid) function for the input value `x0`.

    The log-expit function is defined as::
    
        log_expit(x) = -log(1 + exp(-x))
    
    This function is useful in scenarios where the log-sigmoid is preferred
    for numerical stability or when working with log-probabilities.

    Parameters:
    -----------
    x0 : dfg_number_t
        The input value, which can be a `double`, `float`, or `long double`.

    Returns:
    --------
    dfg_number_t
        The logarithm of the sigmoid of the input value, of the same type as `x0`.

    Example:
    --------
    >>> log_expit(0.5)
    -0.4740769841801067
    """
    return -log(1.0 + exp(-x0))


cpdef dfg_number_t logit(dfg_number_t x0) noexcept nogil:
    """
    Compute the logit function, which is the inverse of the sigmoid function, for the input value `x0`.

    The logit function is defined as::
    
        logit(x) = log(x / (1 - x))
    
    It is commonly used in logistic regression and other statistical models to
    transform probabilities (ranging from 0 to 1) into real-valued numbers.

    Parameters:
    -----------
    x0 : dfg_number_t
        The input value, which should be in the range (0, 1) and can be a 
        `double`, `float`, or `long double`.

    Returns:
    --------
    dfg_number_t
        The logit of the input value, of the same type as `x0`.

    Example:
    --------
    >>> logit(0.5)
    0.0
    """
    return log(x0 / (1.0 - x0))