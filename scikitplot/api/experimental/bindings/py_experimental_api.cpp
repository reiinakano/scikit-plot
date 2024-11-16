// # scikitplot/_experimental/bindings/py_experimental_api.cpp
// C++ Source File
// Pybind11 implementation write C++ directly
// Pybind11 focuses on bridging C++ with Python directly.
// .cpp Files: Pybind11 uses standard C++ source files.
// You write your binding code directly in C++ using the Pybind11 API
// to expose C++ functions and classes to Python.

// scikitplot/
// ├── _experimental/
// │   ├── bindings/                    # Bindings for Pybind11
// │   │   └── py_experimental_api.cpp  # Pybind11 bindings
// │   ├── include/   
// │   ├── NumCpp/                      # Include directory for NumCpp headers
// │   └── src/ 
// │       └── hello.cpp                # Implementation of a greeting function
// ├── CMakeLists.txt or meson.build    # Build configuration
// └── setup.py                         # Setup script for packaging

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For STL bindings
#include <pybind11/numpy.h> // For numpy array bindings

#include "hello.cpp"        // Include header or function prototypes if needed

namespace py = pybind11;


// Expose the functions to Python Module using Pybind11
// In practice, implementation and binding code will generally be located in separate files.
// https://pybind11.readthedocs.io/en/stable/reference.html#c.PYBIND11_MODULE
PYBIND11_MODULE(py_experimental_api, m) {
    m.doc() = R"(
      Experimental API Python module that uses C/C++ for numerical computations.
      Created by Pybind11.
    )"; // optional module docstring
  
    // Add bindings here  
    // Define module functions using a Lambda Function and a docstring
    m.def("py_print_message", 
        [](std::string message = 
           "Hello, from Pybind11 C++!") {
            print_message(message);
        }, 
        py::arg("message") = 
           "Hello, from Pybind11 C++!", 
        "Prints a Unicode message. Default: 'Hello, from Pybind11 C++!'");
}