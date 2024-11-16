// src/py_math.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


// C++ function to calculate the sum of squares
extern "C" double sum_of_squares(const py::array_t<double>& input_array) {
    // Get the buffer information
    py::buffer_info buf_info = input_array.request();
    double* ptr = static_cast<double*>(buf_info.ptr);
    size_t size = buf_info.size;

    // Check for empty array
    if (size == 0) {
        throw std::runtime_error("Input array is empty.");
    }

    // Calculate the sum of squares
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += ptr[i] * ptr[i]; // Square each element and accumulate the sum
    }

    return sum;
}