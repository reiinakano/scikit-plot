// src/py_random.cpp
#include <pybind11/numpy.h>

#include <NumCpp.hpp>

namespace py = pybind11;


// Function to create a random NumPy array from NumCpp
extern "C" py::array_t<double> random_array(int rows, int cols) {
    auto array = nc::random::rand<double>(
        {static_cast<nc::uint32>(rows), 
         static_cast<nc::uint32>(cols)});
  
    py::buffer_info buf_info(
        array.data(), 
        sizeof(double), 
        py::format_descriptor<double>::format(), 
        2, 
        {static_cast<size_t>(rows), 
         static_cast<size_t>(cols)}, 
        {static_cast<size_t>(cols) * sizeof(double), 
         sizeof(double)} 
    );

    return py::array_t<double>(buf_info);
}