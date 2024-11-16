# cy_numcpp_api.pxd
# Cython Declaration File (.pxd) (like C headers) (Optional)
# Purpose: .pxd files serve as Cython declaration files, similar to C header files.
# They are used to declare the structure of C/C++ functions, classes, types,
# and variables that can be shared across multiple .pyx files without redefining them.
# Usage: These files do not contain the actual implementation but make functions
# and variables available to other Cython files. They allow you to declare functions,
# classes, and types that will be used in .pyx files.

# Import the necessary C++ standard library components
from libcpp.string cimport string

# Import reusable code from the .pxi file, If Needed avoid duplicates
# include "cy_numcpp_api.pxi"

# Declare a function from NumCpp header
cdef extern from "src/version.cpp":
    string get_version()  # Declaration of the get_version function

# Declare external C++ functions
cdef extern from "src/hello.cpp":
    void print_message(const char* message)  # Declaration of the print function