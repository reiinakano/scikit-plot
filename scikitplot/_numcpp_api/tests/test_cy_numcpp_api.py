import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst
import ast

# Testing Cython python module implementation
from scikitplot._numcpp_api import cy_numcpp_api


class TestPrint:
    """Test class for C++ print functions."""
    # https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html#setting-capturing-methods-or-disabling-capturing    
    def test_cy_say_hello_inline(self, capsys):  # "capsys" or use "capfd" for fd-level
        # Test the print function
        cy_numcpp_api.py_say_hello_inline()
        # Capture the printed output
        captured = capsys.readouterr()
        # Assert the output is correct
        assert captured.out.strip() == "b'Hello, from Cython .pxi file!'".strip()
      
    def test_cy_print_message(self, capfd):
        """Test to assert 'Hello, from Cython C++!' is printed to stdout."""
        # Call the function that prints the message
        cy_numcpp_api.py_print_message()
        # Capture the printed output at file descriptor level (for C++ integration)
        captured = capfd.readouterr()
        # Check if the output matches the expected message
        assert captured.out.strip() == "Hello, from Cython C++!".strip()


class TestNumCppApi(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)  # Seed for reproducibility
        # Generate a random array for testing
        self.arr_1d = np.random.rand(10)  # 1D array with 10 random values
        self.arr_2d = np.random.rand(4, 3)  # 2D array for testing

    def test_py_random_array(self):
        # This function should be defined if you want to test it
        # Ensure that py_random_array is properly implemented
        random_arr = cy_numcpp_api.py_random_array(3, 4)
        assert random_arr.shape == (3, 4)
        assert np.all(random_arr >= 0) and np.all(random_arr < 1)

        # Generate a single random float
        single_float = cy_numcpp_api.py_random_array()
        assert 0 <= single_float < 1

    def test_py_sum_of_squares_1d(self):
        # Call the sum_of_squares function for the 1D array
        result = cy_numcpp_api.py_sum_of_squares(self.arr_1d)    
        expected_result = np.sum(self.arr_1d ** 2)
        np_testing.assert_almost_equal(result, expected_result, decimal=5)

    def test_py_sum_of_squares_2d_axis0(self):
        # Check sum of squares along axis 0 for the 2D array
        result_2d = cy_numcpp_api.py_sum_of_squares(self.arr_2d, axis=0)
        expected_result_2d = np.sum(self.arr_2d ** 2, axis=0)
        np_testing.assert_almost_equal(result_2d, expected_result_2d, decimal=5)

    def test_py_sum_of_squares_2d_axis1(self):
        # Check sum of squares along axis 1 for the 2D array
        result_2d_axis1 = cy_numcpp_api.py_sum_of_squares(self.arr_2d, axis=1)
        expected_result_2d_axis1 = np.sum(self.arr_2d ** 2, axis=1)
        np_testing.assert_almost_equal(result_2d_axis1, expected_result_2d_axis1, decimal=5)

    def test_py_sum_of_squares_2d_none_axis(self):
        # Check sum of squares for the 2D array with axis=None
        result_2d_none = cy_numcpp_api.py_sum_of_squares(self.arr_2d, axis=None)
        expected_result_2d_none = np.sum(self.arr_2d ** 2)
        np_testing.assert_almost_equal(result_2d_none, expected_result_2d_none, decimal=5)

    def test_invalid_axis(self):
        # Test invalid axis for the 2D array
        with self.assertRaises(ValueError):
            cy_numcpp_api.py_sum_of_squares(self.arr_2d, axis=2)  # Axis out of bounds
        with self.assertRaises(ValueError):
            cy_numcpp_api.py_sum_of_squares(self.arr_2d, axis=-5)  # Negative axis out of bounds


if __name__ == "__main__":
    pytest.main()
    unittest.main()