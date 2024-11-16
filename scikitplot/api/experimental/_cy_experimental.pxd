"""
This module provides efficient implementations of mathematical functions 
using fused types for high precision and performance. It includes functions 
for calculating the expit (logistic sigmoid) function, its logarithm, 
and the logit function.

Fused types are utilized here to allow the same function to be used with
multiple types of numeric inputs, increasing the flexibility and efficiency of the code.

This .pxd file serves as a Cython declaration file,
which is similar in function to header files in C/C++.
It declares C/C++ functions, variables,
and types that will be used in `.pyx` files.
"""

# Define fused types for numeric operations

# number_t: Supports both double and double complex.
ctypedef fused number_t:
    double complex
    double

# Dd_number_t: Supports double and double complex.
ctypedef fused Dd_number_t:
    double complex
    double

# df_number_t: Supports double and float.
ctypedef fused df_number_t:
    double
    float

# dfg_number_t: Extends support to double, float, and long double.
ctypedef fused dfg_number_t:
    double
    float
    long double

# dlp_number_t: Includes double, long, and Py_ssize_t, which is useful for
# working with different numeric types in a variety of contexts.
ctypedef fused dlp_number_t:
    double
    long
    Py_ssize_t

#
# Test function exports
#
# Function declarations with docstrings
# cpdef dfg_number_t expit(dfg_number_t x0) except * nogil
cpdef dfg_number_t expit(dfg_number_t x0) noexcept nogil
cpdef dfg_number_t log_expit(dfg_number_t x0) noexcept nogil
cpdef dfg_number_t logit(dfg_number_t x0) noexcept nogil