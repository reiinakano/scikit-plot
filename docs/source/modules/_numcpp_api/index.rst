.. _numcpp_api:

======================================================================
NumCpp C-API (experimental)
======================================================================

This module contains functions related to :py:mod:`~._numcpp_api`.

.. note::

    The NumCpp API is experimental, and is not yet implemented for any
    functions. Please refer to the :ref:`list of supported and unsupported
    functions <numcpp_api_functions>` for more information. It may change without
    the usual deprecation cycle.
    ::

        >>> import scikitplot as skplt
        >>> dir(skplt._numcpp_api)

NumCpp provides a C-API to enable users to extend the system and get access to the array object for use in other routines. The best way to truly understand the C-API is to read the source code. If you are unfamiliar with (C) source code, however, this can be a daunting experience at first. Be assured that the task becomes easier with practice, and you may be surprised at how simple the C-code can be to understand. Even if you don’t think you can write C-code from scratch, it is much easier to understand and modify already-written source code than create it de novo.

.. seealso::

   * https://numpy.org/doc/stable/reference/c-api/index.html

   * `PyTorch C++ API <https://pytorch.org/cppdocs/>`__.


.. _numcpp_api_functions:

NumCpp Status
----------------------------------------------------------------------

Here is a list of NumCpp functions.

.. jupyter-execute::

    >>> import scikitplot as skplt
    >>> skplt._utils.inspect_module(debug=True)

NumCpp used to below Functions:

- ""


Testing
----------------------------------------------------------------------

**C++ Standards**:

.. image:: https://img.shields.io/badge/C%2B%2B-17-blue.svg
   :target: https://isocpp.org/std/the-standard


.. image:: https://img.shields.io/badge/C%2B%2B-20-blue.svg
   :target: https://isocpp.org/std/the-standard


**Compilers**:

- Visual Studio: 2022  
- GNU: 11.3  
- Clang: 14

**Boost Versions**:  

- 1.73+

.. seealso::

   NumCpp is a templatized header only C++ implementation of the Python :py:mod:`numpy` library.

   Author: David Pilger (dpilger26@gmail.com)

   [1] https://dpilger26.github.io/NumCpp/doxygen/html/index.html

   [2] https://github.com/dpilger26/NumCpp


Developers: From NumPy To NumCpp – A Quick Start Guide
----------------------------------------------------------------------

This guide provides a quick overview of some features of **NumCpp**. Visit the `Full Documentation <https://dpilger26.github.io/NumCpp>`__ for a detailed description.

.. grid:: 1 1 1 1
    :gutter: 1 1 1 1

    .. grid-item-card:: CONTAINERS
        :columns: auto

        The main data structure in NumCpp is the NdArray. It is inherently a 2D array class, with 1D arrays being implemented as 1xN arrays. There is also a DataCube class that is provided as a convenience container for storing an array of 2D NdArrays, but it has limited usefulness past a simple container.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    a = np.array([[1, 2], [3, 4], [5, 6]])
                    a.reshape([2, 3])
                    a.astype(np.double)

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::NdArray<int> a = { {1, 2}, {3, 4}, {5, 6} };
                    a.reshape(2, 3);
                    a.astype<double>();

    .. grid-item-card:: INITIALIZERS
        :columns: auto

        Many initializer functions are provided that return NdArrays for common needs.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    np.linspace(1, 10, 5)
                    np.arange(3, 7)
                    np.eye(4)
                    np.zeros([3, 4])
                    np.ones([3, 4])
                    np.empty([3, 4])
                    np.nan([3, 4])

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::linspace<double>(1, 10, 5);
                    nc::arange<int>(3, 7);
                    nc::eye<double>(4);
                    nc::zeros<double>(3, 4); nc::NdArray<dtype>(3, 4) a = 0;
                    nc::ones<double>(3, 4); nc::NdArray<dtype>(3, 4) a = 1;
                    nc::empty<double>(3, 4); nc::NdArray<dtype>(3, 4) a
                    nc::nans(3, 4); nc::NdArray<double>(3, 4) a = nc::constants::nan;

    .. grid-item-card:: SLICING/BROADCASTING
        :columns: auto

        NumCpp offers NumPy style slicing and broadcasting.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    a[2, 3]
                    a[2:5, 5:8]
                    a[:, 7]
                    a[a > 5]
                    a[a > 5] = 0

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    a(2, 3);
                    a(nc::Slice(2, 5), nc::Slice(5, 8)); a({2, 5}, {5, 8});
                    a(a.rSlice(), 7);
                    a[a > 5];
                    a.putMask(a > 5, 0);

    .. grid-item-card:: RANDOM
        :columns: auto

        The random module provides simple ways to create random arrays.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    np.random.seed(666)
                    np.random.randn(3, 4)
                    np.random.randint(0, 10, [3, 4])
                    np.random.rand(3, 4)
                    np.random.choice(a, 3)

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::random::seed(666);
                    nc::random::randN<double>(nc::Shape(3, 4)); nc::random::randN<double>({3, 4});
                    nc::random::randInt<int>(nc::Shape(3, 4), 0, 10); nc::random::randInt<int>({3, 4}, 0, 10);
                    nc::random::rand<double>(nc::Shape(3, 4)); nc::random::rand<double>({3, 4});
                    nc::random::choice(a, 3);

    .. grid-item-card:: CONCATENATION
        :columns: auto

        Many ways to concatenate NdArray are available.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    np.stack([a, b, c], axis=0)
                    np.vstack([a, b, c])
                    np.hstack([a, b, c])
                    np.append(a, b, axis=1)

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::stack({a, b, c}, nc::Axis::ROW);
                    nc::vstack({a, b, c});
                    nc::hstack({a, b, c});
                    nc::append(a, b, nc::Axis::COL);

    .. grid-item-card:: DIAGONAL, TRIANGULAR, AND FLIP
        :columns: auto

        The following return new NdArrays.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    np.diagonal(a)
                    np.triu(a)
                    np.tril(a)
                    np.flip(a, axis=0)
                    np.flipud(a)
                    np.fliplr(a)

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::diagonal(a);
                    nc::triu(a);
                    nc::tril(a);
                    nc::flip(a, nc::Axis::ROW);
                    nc::flipud(a);
                    nc::fliplr(a);

    .. grid-item-card:: ITERATION
        :columns: auto

        NumCpp follows the idioms of the C++ STL providing iterator pairs to iterate on arrays in different fashions.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    for value in a:
                        pass
                    for value in a.flatten():
                        pass

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    for(auto it = a.begin(); it < a.end(); ++it);
                    for(auto& value : a);

    .. grid-item-card:: LOGICAL FUNCTIONS
        :columns: auto

        Logical FUNCTIONS in NumCpp behave the same as NumPy.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    np.where(a > 5, a, b)
                    np.any(a)
                    np.all(a)
                    np.logical_and(a, b)
                    np.logical_or(a, b)
                    np.isclose(a, b)
                    np.allclose(a, b)

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::where(a > 5, a, b);
                    nc::any(a);
                    nc::all(a);
                    nc::logical_and(a, b);
                    nc::logical_or(a, b);
                    nc::isclose(a, b);
                    nc::allclose(a, b);

    .. grid-item-card:: COMPARISONS
        :columns: auto

        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    np.equal(a, b)
                    np.not_equal(a, b)
                    rows, cols = np.nonzero(a)

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::equal(a, b); a == b;
                    nc::not_equal(a, b); a != b;
                    auto [rows, cols] = nc::nonzero(a);

    .. grid-item-card:: MINIMUM, MAXIMUM, SORTING
        :columns: auto

        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    np.min(a)
                    np.max(a)
                    np.argmin(a)
                    np.argmax(a)
                    np.sort(a, axis=0)
                    np.argsort(a, axis=1)
                    np.unique(a)
                    np.setdiff1d(a, b)
                    np.diff(a)

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::min(a);
                    nc::max(a);
                    nc::argmin(a);
                    nc::argmax(a);
                    nc::sort(a, nc::Axis::ROW);
                    nc::argsort(a, nc::Axis::COL);
                    nc::unique(a);
                    nc::setdiff1d(a, b);
                    nc::diff(a);

    .. grid-item-card:: REDUCERS
        :columns: auto

        Reducers accumulate values of NdArrays along specified axes. When no axis is specified, values are accumulated along all axes.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    np.sum(a); np.sum(a, axis=0);
                    np.prod(a); np.prod(a, axis=0);
                    np.mean(a); np.mean(a, axis=0);
                    np.count_nonzero(a); np.count_nonzero(a, axis=0);

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    nc::sum(a); nc::sum(a, nc::Axis::ROW);
                    nc::prod(a); nc::prod(a, nc::Axis::ROW);
                    nc::mean(a); nc::mean(a, nc::Axis::ROW);
                    nc::count_nonzero(a); nc::count_nonzero(a, nc::Axis::ROW);

    .. grid-item-card:: I/O
        :columns: auto

        Print and file output methods. All NumCpp classes support a print() method and << stream operators.
        ^^^
        .. grid:: 1 1 2 2
            :gutter: 0 0 0 0
            
            .. grid-item-card:: NumPy

                .. code-block:: python

                    print(a)
                    a.tofile(filename, sep='\n')
                    np.fromfile(filename, sep='\n')
                    np.dump(a, filename)
                    np.load(filename)

            .. grid-item-card:: NumCpp

                .. code-block:: python

                    a.print(); std::cout << a;
                    a.tofile(filename, '\n');
                    nc::fromfile<dtype>(filename, '\n')
                    nc::dump(a, filename);
                    nc::load<dtype>(filename);

    .. grid-item-card:: MATHEMATICAL FUNCTIONS
        :columns: auto

        NumCpp universal functions are provided for a large set number of mathematical functions.
        ^^^
        .. grid:: 1 1 1 1
            :gutter: 1 1 1 1
            
            .. grid-item-card:: BASIC FUNCTIONS
                :columns: auto

                .. grid:: 1 1 2 2
                    :gutter: 0 0 0 0
                    
                    .. grid-item-card:: NumPy

                        .. code-block:: python

                            np.abs(a)
                            np.sign(a)
                            np.remainder(a, b)
                            np.clip(a, 3, 8)
                            np.interp(x, xp, fp)

                    .. grid-item-card:: NumCpp

                        .. code-block:: python

                            nc::abs(a);
                            nc::sign(a);
                            nc::remainder(a, b);
                            nc::clip(a, 3, 8);
                            nc::interp(x, xp, fp);

            .. grid-item-card:: EXPONENTIAL FUNCTIONS
                :columns: auto

                .. grid:: 1 1 2 2
                    :gutter: 0 0 0 0
                    
                    .. grid-item-card:: NumPy

                        .. code-block:: python

                            np.exp(a)
                            np.expm1(a)
                            np.log(a)
                            np.log1p(a)

                    .. grid-item-card:: NumCpp

                        .. code-block:: python

                            nc::exp(a);
                            nc::expm1(a);
                            nc::log(a);
                            nc::log1p(a);

            .. grid-item-card:: POWER FUNCTIONS
                :columns: auto

                .. grid:: 1 1 2 2
                    :gutter: 0 0 0 0
                    
                    .. grid-item-card:: NumPy

                        .. code-block:: python

                            np.power(a, 4)
                            np.sqrt(a)
                            np.square(a)
                            np.cbrt(a)

                    .. grid-item-card:: NumCpp

                        .. code-block:: python

                            nc::power(a, 4);
                            nc::sqrt(a);
                            nc::square(a);
                            nc::cbrt(a);

            .. grid-item-card:: TRIGONOMETRIC FUNCTIONS
                :columns: auto

                .. grid:: 1 1 2 2
                    :gutter: 0 0 0 0
                    
                    .. grid-item-card:: NumPy

                        .. code-block:: python

                            np.sin(a); np.sin(b);
                            np.cos(a)
                            np.tan(a)

                    .. grid-item-card:: NumCpp

                        .. code-block:: python

                            nc::sin(a); nc::sin(b);
                            nc::cos(a);
                            nc::tan(a);

            .. grid-item-card:: HYPERBOLIC FUNCTIONS
                :columns: auto

                .. grid:: 1 1 2 2
                    :gutter: 0 0 0 0
                    
                    .. grid-item-card:: NumPy

                        .. code-block:: python

                            np.sinh(a); np.sinh(b);
                            np.cosh(a)
                            np.tanh(a)

                    .. grid-item-card:: NumCpp

                        .. code-block:: python

                            nc::sinh(a); nc::sinh(b);
                            nc::cosh(a);
                            nc::tanh(a);

            .. grid-item-card:: CLASSIFICATION FUNCTIONS
                :columns: auto

                .. grid:: 1 1 2 2
                    :gutter: 0 0 0 0
                    
                    .. grid-item-card:: NumPy

                        .. code-block:: python

                            np.isnan(a)
                            np.isinf(a)

                    .. grid-item-card:: NumCpp

                        .. code-block:: python

                            nc::isnan(a);
                            nc::isinf(a);

            .. grid-item-card:: LINEAR ALGEBRA
                :columns: auto

                .. grid:: 1 1 2 2
                    :gutter: 0 0 0 0
                    
                    .. grid-item-card:: NumPy

                        .. code-block:: python

                            np.linalg.norm(a)
                            np.dot(a, b)
                            np.linalg.det(a)
                            np.linalg.inv(a)
                            np.linalg.lstsq(a, b)
                            np.linalg.matrix_power(a, 3)
                            np.linalg.multi_dot(a, b, c)
                            np.linalg.svd(a)

                    .. grid-item-card:: NumCpp

                        .. code-block:: python

                            nc::norm(a);
                            nc::dot(a, b);
                            nc::linalg::det(a);
                            nc::linalg::inv(a);
                            nc::linalg::lstsq(a, b);
                            nc::linalg::matrix_power(a, 3);
                            nc::linalg::multi_dot({a, b, c});
                            nc::linalg::svd(a);