"""
scikitplot._xp_core_api
========================

This module serves as the core API for the `xp` array namespace,
providing essential functionalities for array operations.

Core Features
--------------
- Create and manipulate arrays.
- Perform mathematical and statistical operations.
- Provide utility functions for array handling and processing.

Behind `xp` functionality
----------------------------

The `_xp_core_api` module offers a simple and intuitive interface for users to interact 
with array data. Below are some key functionalities:

1. **Array Creation**:
   - Create arrays from various data structures, such as lists or tuples.
   - Example: 
     ```python
     import _xp_core_api as xp
     arr = xp.create_array([1, 2, 3, 4])
     ```

2. **Mathematical Operations**:
   - Perform operations like addition, subtraction, multiplication, and division on arrays.
   - Example:
     ```python
     import _xp_core_api as xp
     result = xp.add_arrays(arr, [5, 6, 7, 8])
     ```

3. **Statistical Functions**:
   - Calculate mean, median, standard deviation, etc.
   - Example:
     ```python
     import _xp_core_api as xp
     average = xp.mean(arr)
     ```

4. **Array Manipulation**:
   - Reshape, slice, and transform arrays for various needs.
   - Example:
     ```python
     import _xp_core_api as xp
     reshaped = xp.reshape_array(arr, (2, 2))
     ```

5. **Utilities**:
   - Provide additional utility functions for array processing and handling.
   - Example:
     ```python
     import _xp_core_api as xp
     info = xp.array_info(arr)
     ```

Notes
-----
- Ensure that the `xp` namespace is correctly imported before using the 
  functionalities within this module.
- For more advanced usage, refer to the documentation of the individual 
  functions available in this module.
"""
# scikitplot/_xp_core_lib/__init__.py