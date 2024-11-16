// # scikitplot/_numcpp_api/bindings/py_numcpp_api.cpp
// C++ Source File
// Pybind11 implementation write C++ directly
// Pybind11 focuses on bridging C++ with Python directly.
// .cpp Files: Pybind11 uses standard C++ source files.
// You write your binding code directly in C++ using the Pybind11 API
// to expose C++ functions and classes to Python.

// NumCpp/                            # Include directory for NumCpp headers
// scikitplot/
// ├── _numcpp_api/
// │   ├── bindings/                  # Bindings for Pybind11
// │   │   └── py_numcpp_api.cpp      # Pybind11 bindings
// │   ├── include/   
// │   └── src/  
// │       ├── version.cpp            # Implementation for version retrieval 
// │       ├── hello.cpp              # Implementation of a greeting function
// │       ├── py_math.cpp            # Implementation for mathematical functions
// │       └── py_random.cpp          # Implementation for random number generation
// ├── CMakeLists.txt or meson.build  # Build configuration
// └── setup.py                       # Setup script for packaging

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For STL bindings
#include <pybind11/numpy.h> // For numpy array bindings

#include "version.cpp"      // Include for version
#include "hello.cpp"        // Include header or function prototypes if needed
#include "py_math.cpp"      // Include header or function prototypes if needed
#include "py_random.cpp"    // Include header or function prototypes if needed

namespace py = pybind11;


// Expose the functions to Python Module using Pybind11
// In practice, implementation and binding code will generally be located in separate files.
// https://pybind11.readthedocs.io/en/stable/reference.html#c.PYBIND11_MODULE
PYBIND11_MODULE(py_numcpp_api, m) {
  m.doc() = 
    R"(\
NumCpp API Python module that uses Pybind11 C/C++ for numerical computations.
Created by Pybind11.
)"; // optional module docstring
  
  // https://github.com/jupyter/jupyter-sphinx

  // Add bindings here
  // Expose the VERSION constant to Python as a function that returns it
  // Return the version string defined in Version.hpp
  m.def("py_get_version",
    []() { return nc::VERSION; },
    R"(\
Get the NumCpp header library version.

Returns
-------
str
    NumCpp header library version.

Examples
--------
.. jupyter-execute::

    >>> from scikitplot import _numcpp_api as numcpp
    >>> numcpp.__version__
)"
  );

  // Define module functions using a Lambda Function and a docstring
  m.def("py_print_message",
    [](std::string message =
      "Hello, from Pybind11 C++!") { print_message(message); },
    py::arg("message") = "Hello, from Pybind11 C++!",
    R"(\
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

    >>> from scikitplot import _numcpp_api as numcpp
    >>> numcpp.py_numcpp_api.py_print_message()
)"
  );

  // Define module functions using a Function Pointer and a docstring
  m.def("py_sum_of_squares",
    &sum_of_squares,
    R"(\
Calculate the sum of squares.

Parameters
----------
arg0 : array-like

Returns
-------
float
    Sum of squares.

Examples
--------
.. jupyter-execute::

    >>> from scikitplot import _numcpp_api as numcpp
    >>> import numpy as np; np.random.seed(0)
    >>> arr = np.array([1,2])
    >>> numcpp.py_numcpp_api.py_sum_of_squares(arr)
)"
  );
  
  // Define module functions using a Function Pointer and a docstring
  m.def("py_random_array",
    &random_array,
    R"(\
Create a random NumCpp array.

Parameters
----------
arg0 : int
    Row.
    
arg1 : int
    Col.

Returns
-------
numpy.array
    2D array-like, shape (arg0, arg1)

Examples
--------
.. jupyter-execute::

    >>> from scikitplot import _numcpp_api
    >>> import numpy as np; np.random.seed(0)
    >>> arr = _numcpp_api.py_numcpp_api.py_random_array(1, 2)
    >>> arr
)"
  );
}