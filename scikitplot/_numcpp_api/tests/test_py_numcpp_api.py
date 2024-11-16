import numpy as np
import numpy.testing as np_testing
import pytest
import unittest
import hypothesis
import hypothesis.extra.numpy as npst

# Testing pybind11 python module implementation
from scikitplot._numcpp_api import py_numcpp_api


class TestPrint:
    """Test class for py_print_message."""
    # https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html#setting-capturing-methods-or-disabling-capturing
    def test_py_print_message(self, capfd):  # "capsys" or use "capfd" for fd-level
        """Test to assert 'Hello, from Pybind11 C++!' is printed to stdout."""
        # Call the function that prints the message
        py_numcpp_api.py_print_message()

        # Capture the printed output at file descriptor level (for C++ integration)
        captured = capfd.readouterr()

        # Check if the output matches the expected message
        assert captured.out.strip() == "Hello, from Pybind11 C++!".strip()


class TestNumCppApi(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)  # Seed for reproducibility
      
    def test_py_random_array(self):
        """Test to generate a random NumPy array with specified dimensions."""
        # Call the random_array function with specific dimensions
        rows, cols = 3, 4
        arr = py_numcpp_api.py_random_array(rows, cols)
        
        # Check the shape of the returned array
        assert arr.shape == (rows, cols)
        
        # Check the data type of the elements in the array
        assert arr.dtype == np.float64
    
        # Check that the array contains the expected number of elements
        assert arr.size == rows * cols
    
        # Check that the values are within the expected range (0 to 1 for random)
        assert np.all(arr >= 0) and np.all(arr < 1)

    def test_sum_of_squares(self):
        """Test the sum_of_squares function."""
        # Create a sample array
        sample_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        # Calculate the expected sum of squares
        expected_result = np.sum(sample_arr ** 2)
        
        # Call the sum_of_squares function
        result = py_numcpp_api.py_sum_of_squares(sample_arr)
        
        # Assert the result is as expected
        np_testing.assert_almost_equal(result, expected_result, decimal=5)

    def test_sum_of_squares_empty_array(self):
        """Test to ensure that an empty array raises an error."""
        empty_arr = np.array([], dtype=np.float64)
        
        # Check that an exception is raised for an empty array
        with pytest.raises(RuntimeError, match="Input array is empty."):
            py_numcpp_api.py_sum_of_squares(empty_arr)


if __name__ == "__main__":
    pytest.main()
    unittest.main()