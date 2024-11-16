"""
NumCpp API Python module that uses Ceyton C/C++ for numerical computations.
Created by Cython.
"""
# cy_numcpp_api.pyx
# Cython Implementation File (.pyx) (main logic) (Python-like syntax)
# Purpose: .pyx files are the primary Cython implementation files.
# You write Cython code, which is a mix of Python and C, in these files.
# Usage: These files can contain both Python code and Cython code
# for performance-critical sections. They allow you to write high-performance code
# that is still very readable and can easily interface with Python. When compiled,
# .pyx files yield C/C++ extensions that can be imported and used in Python.

import numpy as np  # Import Cython's NumPy support
cimport numpy as cnp  # Cython's NumPy support

# Ensure NumPy is properly initialized for Cython
cnp.import_array()

from .cy_numcpp_api cimport *  # Import declarations from .pxd
include "cy_numcpp_api.pxi"    # Include the .pxi file, If Needed avoid duplicates


## Define __all__ to specify the public interface of the module,
# not required default all above func
__all__ = [
  'py_get_version',
  'py_print_message',
  'py_say_hello_inline',
  'py_random_array',
  'py_sum_of_squares',
]


# Expose the C++ print_message function to Python
def py_get_version():
    """
    Get the NumCpp header library version.
    
    Returns
    -------
    str
        NumCpp header library version.
    
    Examples
    --------
    .. jupyter-execute::
    
        >>> from scikitplot import _numcpp_api
        >>> _numcpp_api.cy_numcpp_api.py_get_version()
    """
    # Call the C++ function and return the version (utf-8 decoded bytes)
    return get_version().decode('utf-8')


# Expose the C++ print_message function to Python
def py_print_message(message="Hello, from Cython C++!"):
    """
    Prints a Unicode message.
    
    Parameters
    ----------
    message : str, optional, default='Hello, from Pybind11 C++!'
        Prints a Unicode message.
    
    Returns
    -------
    None
        Prints a Unicode message.
    
    Examples
    --------    
    .. jupyter-execute::
    
        >>> from scikitplot import _numcpp_api 
        >>> _numcpp_api.cy_numcpp_api.py_print_message()
    """
    print_message(message.encode('utf-8'))  # Call the C++ function with the message


# Expose the C++ inline function to Python
def py_say_hello_inline(message="Hello, from Cython .pxi file!"):
    """
    Prints a unicode message as byte.
    
    Parameters
    ----------
    message : str, optional, default='Hello, from Pybind11 C++!'
        Prints a unicode message as byte.
    
    Returns
    -------
    None
        Prints a unicode message as byte.
    
    Examples
    --------    
    .. jupyter-execute::
    
        >>> from scikitplot import _numcpp_api 
        >>> _numcpp_api.cy_numcpp_api.py_say_hello_inline()
    """
    # Call the C++ inline function, ensuring the Python string is converted to char* (utf-8 encoded bytes)
    say_hello_inline(message.encode('utf-8'))#.decode('utf-8')


# Cython function interfacing with C++
# Python-exposed function to calculate the sum of squares of a NumPy array
def py_sum_of_squares(
    cnp.ndarray arr,
    axis=None,
    dtype=None,
    out=None,
    keepdims=np._NoValue,
    initial=np._NoValue,
    where=np._NoValue
):
    """
    Calculate the sum of squares of a NumPy array along a specified axis.

    Parameters
    ----------
    arr : ndarray
        Input array (can be of any dimension).
    axis : None or int or tuple of ints, optional
        The axis or axes along which the sum of squares is computed.
        By default (axis=None), the sum is computed over the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which
        the elements are summed. If not provided, defaults to the dtype of arr.
    out : ndarray, optional
        Alternative output array in which to place the result.
        It must have the same shape as the expected output.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. This option allows for correct
        broadcasting against the input array.
    initial : scalar, optional
        Starting value for the sum.
    where : array_like, optional
        Elements to include in the sum.

    Returns
    -------
    int, float or numpy.ndarray
        The sum of squares along the specified axis. If axis is None,
        the sum is computed over all elements.

    Examples
    --------    
    .. jupyter-execute::
    
        >>> from scikitplot import _numcpp_api
        >>> import numpy as np; np.random.seed(0)
        >>> arr = np.array([1,2])
        >>> _numcpp_api.cy_numcpp_api.py_sum_of_squares(arr)
    """
    cdef int n_dim = arr.ndim
    
    if axis is not None:
        # Handle negative axis, making it relative to the last dimension
        if axis < 0:
            axis += n_dim
            
        # Validate axis range
        if axis < 0 or axis >= n_dim:
            raise ValueError(f"Invalid axis: {axis}. Array has {n_dim} dimensions.")

    # Calculate and return the sum of squares along the specified axis
    return np.sum(arr ** 2, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)


# Python-exposed function to get random_array of a NumPy array
def py_random_array(*shape):
    """
    Generate a random NumPy array of the given shape with random values in the range [0, 1).

    Parameters
    ----------
    *shape : int, optional
        The dimensions of the returned array. If no argument is given, 
        a single random float is returned instead.

    Returns
    -------
    float or numpy.ndarray
        - If one or more dimensions are specified, a NumPy array of random values with the given shape is returned.
        - If no dimensions are specified, a single random float is returned.

    Examples
    --------
    Generate a 2D array with shape (3, 4) filled with random values:
    
    .. jupyter-execute::
    
        >>> from scikitplot import _numcpp_api
        >>> import numpy as np; np.random.seed(0)
        >>> arr = _numcpp_api.cy_numcpp_api.py_random_array(3, 4)
        >>> arr    

    Generate a 3D array with shape (2, 3, 4) filled with random values:
    
    .. jupyter-execute::
    
        >>> from scikitplot import _numcpp_api
        >>> import numpy as np; np.random.seed(0)
        >>> arr = _numcpp_api.cy_numcpp_api.py_random_array(2, 3, 4)
        >>> arr

    Generate a single random float:
    
    .. jupyter-execute::
    
        >>> from scikitplot import _numcpp_api
        >>> import numpy as np; np.random.seed(0)
        >>> arr = _numcpp_api.cy_numcpp_api.py_random_array()
        >>> arr
    """    
    # If no shape is provided, return a single random float
    if not shape:
        return np.random.rand()
    
    # Use NumPy to create a random array with the given shape
    return np.random.rand(*shape)