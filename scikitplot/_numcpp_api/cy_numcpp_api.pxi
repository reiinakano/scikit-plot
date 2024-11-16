# cy_numcpp_api.pxi
# Cython Include File (.pxi) (like C headers but for Cython) (Optional)
# Purpose: .pxi files act like include files (or code snippets)
# that can be reused in multiple Cython modules.
# They allow you to "include" common code
# that is shared across several .pyx or .pxd files.
# Usage: This is useful for including helper functions, common code logic,
# or any reusable code snippets in different modules without duplicating code.
# It helps maintain clean and modular code organization in your Cython projects.

# Usage: Defined common functions, classes, or C declarations in a .pxi file
# include "cy_numcpp_api.pxi"

# Include common functions or variables that can be reused in .pxd/.pyx files

# Import the necessary C++ standard library components
from libcpp.string cimport string

# We use inline to reduce function call overhead
# and improve performance by embedding the function directly at the call site.
# Declare the inline function with the correct C++ string handling
cdef inline void say_hello_inline(const char* message = b"Hello, from Cython .pxi file!"):
    # Convert the char* to a C++ string
    cdef string cpp_message = string(message)
    
    # Print the message
    # print(<char *> message.decode('utf-8'))  # Print as a string
    print(cpp_message)
