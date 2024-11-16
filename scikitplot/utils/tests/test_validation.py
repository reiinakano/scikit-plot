import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl

from ..validation import (
  validate_plotting_decorator,
  validate_plotting_kwargs,
  validate_plotting_kwargs_decorator,
  validate_shapes,
  validate_shapes_decorator,
  validate_y_true,
  validate_y_true_decorator,
  validate_y_probas,
  validate_y_probas_decorator,
  validate_y_probas_bounds,
  validate_y_probas_bounds_decorator,
  validate_inputs,
)


# Assume validate_plotting_kwargs and validate_plotting_decorator are already imported
class TestValidatePlottingKwargs:
    """Test suite for validate_plotting_kwargs and validate_plotting_kwargs_decorator"""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Set up the test environment."""
        np.random.seed(0)
        plt.close("all")

    @pytest.fixture(autouse=True)
    def tearDown(self):
        """Set up the test environment."""
        plt.close("all")
        
    def dummy_function(self, *args, **kwargs):
        """
        A method that demonstrates the use of the validate_plotting_kwargs_decorator.
        
        This method wraps another function in the validate_plotting_kwargs_decorator
        to validate the (fig, ax) before returning them.
        
        Parameters:
        - fig: The figure object to be validated (optional).
        - ax: The axes object to be validated (optional).
        - figsize: The size of the figure (optional).
    
        Returns:
        - Tuple of (fig, ax) if the inputs are valid.
        """
        
        @validate_plotting_kwargs_decorator
        def inner_dummy_function(*args, fig=None, ax=None, figsize=None, nrows=1, ncols=1, index=1, **kwargs):
            """
            A dummy plotting function that demonstrates the use of the validate_plotting_kwargs_decorator.
            
            This function is decorated to validate its inputs before execution.
            
            Parameters:
          - fig: The figure object to be validated (optional).
          - ax: The axes object to be validated (optional).
          - figsize: The size of the figure (optional).
            
            Returns:
            - Tuple of (fig, ax) if the inputs are valid.
            """
            # Return the provided figure and axes objects
            return fig, ax
        
        # Call the inner function with the provided arguments
        return inner_dummy_function(*args, **kwargs)

    # Test case 1: No fig or ax provided (should create new ones)
    def test_no_fig_no_ax(self):
        """Test case: No fig or ax provided (should create new ones)."""
        fig, ax = self.dummy_function()
        assert isinstance(fig, mpl.figure.Figure), "Expected a Figure object to be created"
        assert isinstance(ax, mpl.axes.Axes), "Expected an Axes object to be created"

    # Test case 2: Only fig provided (should add subplot to fig)
    def test_fig_only(self):
        """Test case: Only fig provided (should add subplot to fig)."""
        fig = plt.figure()
        new_fig, _ = self.dummy_function(fig=fig)
        assert isinstance(new_fig, mpl.figure.Figure), "Expected a Figure object to be returned"

    # Test case 3: Only ax provided (should use the provided ax)
    def test_ax_only(self):
        """Test case: Only ax provided (should use the provided ax)."""
        fig, ax = plt.subplots()
        new_fig, new_ax = self.dummy_function(ax=ax)
        assert new_fig == fig, "Expected the original Figure to be returned"
        assert new_ax == ax, "Expected the original Axes to be used"

    # Test case 4: Both fig and ax provided (should use both)
    def test_fig_and_ax(self):
        """Test case: Both fig and ax provided (should use both)."""
        fig, ax = plt.subplots()
        new_fig, new_ax = self.dummy_function(fig=fig, ax=ax)
        assert new_fig == fig, "Expected the provided Figure to be returned"
        assert new_ax == ax, "Expected the provided Axes to be used"

    # Test case 5: Invalid ax type (should raise ValueError)
    def test_invalid_ax(self):
        """Test case: Invalid ax type (should raise ValueError)."""
        with pytest.raises(ValueError, match="Provided ax must be an instance of matplotlib.axes.Axes"):
            self.dummy_function(ax="invalid_ax")

    # Test case 6: Invalid fig type (should raise ValueError)
    def test_invalid_fig(self):
        """Test case: Invalid fig type (should raise ValueError)."""
        with pytest.raises(ValueError, match="Provided fig must be an instance of matplotlib.figure.Figure"):
            self.dummy_function(fig="invalid_fig")

    # Test case 7: Decorator test (valid case)
    def test_decorator_creates_fig_and_ax(self):
        """Test case: Decorator creates fig and ax (valid case)."""
        fig, ax = self.dummy_function()
        assert isinstance(fig, mpl.figure.Figure), "Expected a Figure object to be created by the decorator"
        assert isinstance(ax, mpl.axes.Axes), "Expected an Axes object to be created by the decorator"

    # Test case 8: Decorator with provided fig and ax
    def test_decorator_with_fig_and_ax(self):
        """Test case: Decorator with provided fig and ax."""
        fig, ax = plt.subplots()
        new_fig, new_ax = self.dummy_function(fig=fig, ax=ax)
        assert new_fig == fig, "Expected the provided Figure to be used by the decorator"
        assert new_ax == ax, "Expected the provided Axes to be used by the decorator"


# Assume validate_shapes and validate_shapes_decorator are already imported
class TestValidateShapes:
    """Test suite for validate_shapes and validate_shapes_decorator"""

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        plt.close("all")
        
    def dummy_function(self, *args, **kwargs):
        """
        A method that demonstrates the use of the validate_shapes_decorator.
        
        This method wraps another function in the validate_shapes_decorator
        to validate the shapes of y_true and y_probas before returning them.
        
        Parameters:
        - y_true: The true labels (1D or 2D array-like).
        - y_probas: The predicted probabilities (1D or 2D array-like).
    
        Returns:
        - Tuple of y_true and y_probas if shapes are valid.
        """
        
        @validate_shapes_decorator
        def inner_dummy_function(y_true, y_probas, *args, **kwargs):
            """
            Inner function that simply returns y_true and y_probas.
            
            This function is decorated to validate its inputs before execution.
            
            Parameters:
            - y_true: The true labels (1D or 2D array-like).
            - y_probas: The predicted probabilities (1D or 2D array-like).
            
            Returns:
            - Tuple of y_true and y_probas.
            """
            return y_true, y_probas
        
        # Call the inner function with the provided arguments
        return inner_dummy_function(*args, **kwargs)

    # Test case 1: Valid binary case, 1D y_true and y_probas (probability for one class)
    def test_valid_binary_1d(self):
        """Test case: Valid binary case, 1D y_true and y_probas (probability for one class)."""
        y_true = [0, 1, 0, 1]
        y_probas = [0.1, 0.9, 0.2, 0.8]
        try:
            self.dummy_function(y_true, y_probas)
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError: {str(e)}")

    # Test case 2: Valid binary case, 1D y_true and 2D y_probas (probabilities for two classes)
    def test_valid_binary_2d(self):
        """Test case: Valid binary case, 1D y_true and 2D y_probas (probabilities for two classes)."""
        y_true = [0, 1, 0, 1]
        y_probas = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]
        try:
            self.dummy_function(y_true, y_probas)
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError: {str(e)}")

    # Test case 3: Valid multi-class case, 2D y_true and y_probas
    def test_valid_multiclass(self):
        """Test case: Valid multi-class case, 2D y_true and y_probas."""
        y_probas = [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
        
        y_true = [0, 1, 2]
        try:
            self.dummy_function(y_true, y_probas)
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError: {str(e)}")
            
        y_true_bin = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        try:
            self.dummy_function(y_true_bin, y_probas)
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError: {str(e)}")

    # Test case 4: Invalid shapes, mismatch between y_true and y_probas lengths (1D case)
    def test_shape_mismatch_1d(self):
        """Test case: Invalid shapes between y_true and y_probas (should raise ValueError)."""
        y_true = [0, 1, 0]
        y_probas = [0.1, 0.9, 0.2, 0.8]  # Different length
        with pytest.raises(ValueError, match="Shape mismatch"):
            self.dummy_function(y_true, y_probas)

    # Test case 5: Invalid shapes, mismatch between y_true and y_probas dimensions (2D case)
    def test_shape_mismatch_2d(self):
        """Test case: Invalid shapes between y_true and y_probas (should raise ValueError)."""
        y_true = [0, 1, 0]
        y_probas = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]  # Different length
        with pytest.raises(ValueError, match="Shape mismatch"):
            self.dummy_function(y_true, y_probas)

    # Test case 6: Invalid class count in y_probas compared to y_true (multiclass case)
    def test_class_count_mismatch(self):
        """Test case: Class count mismatch (should raise ValueError)."""
        y_true = [[1, 0], [0, 1], [1, 0]]  # Two classes
        y_probas = [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]  # Three classes
        with pytest.raises(ValueError, match="Number of classes in `y_true`"):
            self.dummy_function(y_true, y_probas)

    # Test case 7: Invalid y_true shape
    def test_invalid_y_true_shape(self):
        """Test case: Invalid y_true shape (should raise ValueError)."""
        y_true = [[[0, 1]]]  # Invalid shape
        y_probas = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]
        with pytest.raises(ValueError, match="Invalid shape for `y_true`"):
            self.dummy_function(y_true, y_probas)

    # Test case 8: Decorator test (valid case)
    def test_decorator_valid_case(self):
        """Test case: Decorator creates y_true, y_probas (valid case)."""
        y_true = [0, 1, 0, 1]
        y_probas = [0.1, 0.9, 0.2, 0.8]
        try:
            y_true, y_probas = self.dummy_function(y_true, y_probas)
            assert len(y_true) == len(y_probas), "Decorator should have validated the shapes"
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError: {str(e)}")

    # Test case 9: Decorator test (invalid case)
    def test_decorator_invalid_case(self):
        """Test case: Decorator with provided y_true, y_probas."""
        y_true = [0, 1, 0]
        y_probas = [0.1, 0.9, 0.2, 0.8]  # Length mismatch
        with pytest.raises(ValueError, match="Shape mismatch"):
            self.dummy_function(y_true, y_probas)


class TestValidateYTrue:
    """Test suite for validate_y_true and validate_y_true_decorator"""

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        plt.close("all")
        
    def dummy_function(self, *args, **kwargs):
        """A dummy function to test the decorator."""
        
        @validate_y_true_decorator
        def inner_dummy_function(y_true, *args, pos_label=None, class_index=None, **kwargs):
            """A dummy function to test the decorator."""
            return y_true  # Simply return y_true after validation
        
        # Call the inner function with the provided arguments
        return inner_dummy_function(*args, **kwargs)

    # Test case 1: Valid binary case with default pos_label
    def test_valid_binary_default_pos_label(self):
        y_true = [0, 1, 1, 0]
        expected_output = np.array([False,  True,  True, False])
        result = self.dummy_function(y_true)
        np.testing.assert_array_equal(result, expected_output)

    # Test case 2: Valid binary case with specified pos_label
    def test_valid_binary_specified_pos_label(self):
        y_true = [0, 1, 1, 0]
        expected_output = np.array([True, False, False, True])
        result = self.dummy_function(y_true, pos_label=0)
        np.testing.assert_array_equal(result, expected_output)

    # Test case 3: Valid multi-class case
    def test_valid_multiclass(self):
        y_true = ['class_0', 'class_1', 'class_2', 'class_0']
        expected_output = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # One-hot encoded
        result = self.dummy_function(y_true)
        np.testing.assert_array_equal(result, expected_output)

    # Test case 4: Invalid y_true (non-iterable)
    def test_invalid_y_true_non_iterable(self):
        with pytest.raises(ValueError, match="`y_true` must be of type bool, str, numeric, or a mix"):
            self.dummy_function(12345)  # Invalid input

    # Test case 5: Invalid y_true (not enough distinct classes)
    def test_invalid_y_true_not_enough_classes(self):
        with pytest.raises(ValueError, match="`y_true` must contain more than one distinct class."):
            self.dummy_function([1, 1, 1])  # Only one class

    # Test case 6: Invalid pos_label
    def test_invalid_pos_label(self):
        y_true = [0, 1, 1, 0]
        with pytest.raises(ValueError, match="`pos_label` must be one of label classes:"):
            self.dummy_function(y_true, pos_label=2)  # Invalid pos_label

    # Test case 7: Invalid class_index
    def test_invalid_class_index(self):
        y_true = ['class_0', 'class_1', 'class_2', 'class_0']
        with pytest.raises(ValueError, match="class_index 5 out of bounds for `y_true`. It must be between 0 and 2."):
            self.dummy_function(y_true, class_index=5)  # Invalid class index

    # Test case 8: Decorator test (valid case)
    def test_decorator_valid_case(self):
        y_true = [0, 1, 1, 0]
        expected_output = np.array([False, True, True, False])
        result = self.dummy_function(y_true)
        np.testing.assert_array_equal(result, expected_output)

    # Test case 9: Decorator test (invalid case)
    def test_decorator_invalid_case(self):
        with pytest.raises(ValueError, match="`y_true` must be of type bool, str, numeric, or a mix"):
            self.dummy_function(12345)  # Invalid input


class TestValidateYProbas:
    """Test suite for validate_y_probas and validate_y_probas_decorator"""

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        plt.close("all")

    def dummy_function(self, *args, **kwargs):
        """A dummy function to test the decorator."""
        
        @validate_y_probas_decorator
        def inner_dummy_function(y_true, y_probas, *args, class_index=None, **kwargs):
            """A dummy function to test the decorator."""
            # This function should do something with y_true and the validated y_probas
            # For testing, we can just return y_probas to see what we get
            return y_probas  # Simply return y_probas after validation
        
        # Call the inner function with the provided arguments
        return inner_dummy_function(*args, **kwargs)

    # Test case 1: Valid 1D probabilities
    def test_valid_1d_probas(self):
        y_true = [0, 1]
        y_probas = [0.9, 0.1]
        expected_output = np.array([0.9, 0.1])
        result = self.dummy_function(y_true, y_probas)
        np.testing.assert_array_equal(result, expected_output)

    # Test case 2: Valid 2D probabilities with class_index
    def test_valid_2d_probas_with_class_index(self):
        y_true = [0, 1]
        y_probas = [[0.8, 0.2], [0.2, 0.8]]
        expected_output = np.array([0.2, 0.8])  # Class index 1
        result = self.dummy_function(y_true, y_probas, class_index=1)
        np.testing.assert_array_equal(result, expected_output)

    # Test case 3: Valid 2D probabilities without class_index
    def test_valid_2d_probas_without_class_index(self):
        y_true = [0, 1]
        y_probas = [[0.6, 0.4], [0.3, 0.7]]
        expected_output = np.array([[0.6, 0.4], [0.3, 0.7]])
        result = self.dummy_function(y_true, y_probas)
        np.testing.assert_array_equal(result, expected_output)

    # Test case 4: Invalid y_probas (not 1D or 2D)
    def test_invalid_y_probas_shape(self):
        y_true = [0, 1]
        with pytest.raises(ValueError, match="`y_probas` must be either a 1D or 2D array."):
            self.dummy_function(y_true, np.random.rand(3, 3, 3))  # Invalid shape

    # Test case 5: Invalid class_index
    def test_invalid_class_index(self):
        y_true = [0, 1]
        y_probas = [[0.8, 0.2], [0.2, 0.8]]
        with pytest.raises(ValueError, match="class_index 5 out of bounds for `y_probas`. It must be between 0 and 1."):
            self.dummy_function(y_true, y_probas, class_index=5)  # Invalid class index

    # Test case 6: Decorator test (valid case)
    def test_decorator_valid_case(self):
        y_true = [0, 1]
        y_probas = [[0.9, 0.1], [0.2, 0.8]]
        expected_output = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = self.dummy_function(y_true, y_probas)
        np.testing.assert_array_equal(result, expected_output)

    # Test case 7: Decorator test (invalid case)
    def test_decorator_invalid_case(self):
        y_true = [0, 1]
        with pytest.raises(ValueError, match="`y_probas` must be an array of numerical values."):
            self.dummy_function(y_true, "invalid_input")  # Invalid input


class TestValidateYProbasBounds:
    """Test suite for the `validate_y_probas_bounds` function and its decorator"""

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        plt.close("all")

    def dummy_function(self, *args, **kwargs):
        """A dummy function to test the decorator"""
        
        @validate_y_probas_bounds_decorator
        def inner_dummy_function(y_probas, *args, method="minmax", axis=0, **kwargs):
            """Returns the validated y_probas"""
            return y_probas  # Return y_probas after validation
        
        # Call the inner function with the provided arguments
        return inner_dummy_function(*args, **kwargs)

    # Test case 1: Valid 1D array with scaling
    def test_valid_1d_minmax_scaling(self):
        y_probas = np.array([-0.5, 0.2, 1.5])
        expected_output = np.array([0.0, 0.35, 1.0])
        result = self.dummy_function(y_probas)
        np.testing.assert_array_almost_equal(result, expected_output)

    # Test case 2: Valid 2D array with scaling
    def test_valid_2d_minmax_scaling(self):
        y_probas = np.array([[-5, 0], [0, 0], [10, 15]])
        expected_output = np.array([[0.0, 0.0], [0.333333, 0], [1.0, 1.0]])
        result = self.dummy_function(y_probas)
        np.testing.assert_array_almost_equal(result, expected_output)

    # Test case 3: Valid 1D array with sigmoid scaling
    def test_valid_1d_sigmoid_scaling(self):
        y_probas = np.array([-1, 0, 1])
        expected_output = np.array([0.26894142, 0.5, 0.73105858])
        result = self.dummy_function(y_probas, method="sigmoid")
        np.testing.assert_array_almost_equal(result, expected_output)

    # Test case 4: Valid 2D array with sigmoid scaling
    def test_valid_2d_sigmoid_scaling(self):
        y_probas = np.array([[-5, 0], [10, 15]])
        expected_output = 1 / (1 + np.exp(-y_probas))
        result = self.dummy_function(y_probas, method="sigmoid")
        np.testing.assert_array_almost_equal(result, expected_output)

    # Test case 5: No scaling needed (within bounds)
    def test_no_scaling_needed(self):
        y_probas = np.array([[0.2, 0.5], [0.7, 0.8]])  # Already in [0, 1] range
        expected_output = np.array([[0.2, 0.5], [0.7, 0.8]])
        result = self.dummy_function(y_probas)
        np.testing.assert_array_almost_equal(result, expected_output)

    # Test case 6: Invalid input type
    def test_invalid_input_type(self):
        y_probas = "invalid_input"
        with pytest.raises(ValueError, match="`y_probas` must be an array of numerical values."):
            self.dummy_function("invalid_input")