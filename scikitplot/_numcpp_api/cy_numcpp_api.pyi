# cy_numcpp_api.pyi
# Cython Stub File (.pyi) (like Python type hints for Cython) (Optional)
# Purpose: .pyi files are used for type hinting and provide
# a pure Python interface for Cython modules.
# They serve a similar purpose to Python type hints but specifically for Cython.
# Usage: These files contain the declarations of classes and functions
# without implementation details, allowing tools like type checkers (e.g., mypy)
# or IDEs to understand the types and signatures of the functions
# and classes in the corresponding .pyx files.

cdef void print_message(const std::string& message = "Hello, from C++!")  # Stub for print_message function
cdef double sum_of_squares(const double* arr, long size)  # Stub for sum_of_squares function

def py_sum_of_squares(arr: np.ndarray) -> float:  # Stub for Python-exposed function
    pass  # Implementation not included in .pyi
