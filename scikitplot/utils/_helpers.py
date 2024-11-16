"""
Helper functions and generic utilities for use in scikitplot code.

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import numpy as np
from sklearn.preprocessing import LabelEncoder


## Define __all__ to specify the public interface of the module,
## not required default all belove func
__all__ = [
    'validate_labels',
    'cumulative_gain_curve',
    'binary_ks_curve',
]


def validate_labels(
    known_classes, 
    passed_labels, 
    argument_name,
):
    """
    Validates the labels passed into arguments such as `true_labels` or `pred_labels`
    in functions like `plot_confusion_matrix`.

    This function checks for any duplicate labels and ensures that all passed labels
    are within the set of known classes. It raises a `ValueError` if any issues are found.

    Parameters
    ----------
    known_classes : array-like
        The set of classes that are known to appear in the data.

    passed_labels : array-like
        The labels that were passed in through the argument to be validated.

    argument_name : str
        The name of the argument being validated. Used for error messages.

    Raises
    ------
    ValueError
        If there are duplicate labels in `passed_labels` or if any labels
        in `passed_labels` are not found in `known_classes`.

    Examples
    --------
    >>> known_classes = ["A", "B", "C"]
    >>> passed_labels = ["A", "B"]
    >>> validate_labels(known_classes, passed_labels, "true_labels")
    """
    known_classes = np.array(known_classes)
    passed_labels = np.array(passed_labels)

    unique_labels, unique_indexes = np.unique(passed_labels, return_index=True)

    # Check for duplicates in passed labels
    if len(passed_labels) != len(unique_labels):
        indexes = np.arange(0, len(passed_labels))
        duplicate_indexes = indexes[~np.in1d(indexes, unique_indexes)]
        duplicate_labels = [str(x) for x in passed_labels[duplicate_indexes]]
        raise ValueError(
            "The following duplicate labels were "
            f"passed into {argument_name}: {', '.join(duplicate_labels)}"
        )

    # Check for labels in passed_labels that are not in known_classes
    passed_labels_absent = ~np.in1d(passed_labels, known_classes)
    if np.any(passed_labels_absent):
        absent_labels = [str(x) for x in passed_labels[passed_labels_absent]]
        raise ValueError(
            f"The following labels were passed into {argument_name}, "
            f"but were not found in labels: {', '.join(map(str, absent_labels))}"
        )
    return None


def cumulative_gain_curve(
    y_true, 
    y_score, 
    pos_label=None
):
    """
    Generate the data points necessary to plot the Cumulative Gain curve for binary classification tasks.

    The Cumulative Gain curve helps in visualizing how well a binary classifier identifies the positive class 
    as more instances are considered based on predicted scores. It shows the proportion of true positives 
    captured as a function of the total instances considered.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels of the data. This array should contain exactly two unique classes 
        (e.g., `[0, 1]`, `[-1, 1]`) to represent a binary classification problem. If more 
        than two classes are present, the function will raise a `ValueError`.

    y_score : array-like of shape (n_samples,)
        Target scores for each instance. These scores are typically the predicted probability of the positive 
        class, confidence scores, or any non-thresholded metric produced by a classifier. 
        It is essential that these scores be continuous rather than binary (0/1).

    pos_label : int or str, optional, default=None
        The label representing the positive class. If `pos_label` is not provided, the function 
        attempts to infer it from `y_true`, assuming standard binary classification labels such as 
        `{0, 1}`, `{-1, 1}`, or a single unique class. If inference is not possible, a `ValueError` 
        is raised.

    Returns
    -------
    percentages : numpy.ndarray of shape (n_points,)
        The X-axis values representing the cumulative percentage of instances considered. 
        Values range from 0 (no instances considered) to 1 (all instances considered), and an 
        initial 0 is inserted at the start to represent the baseline.

    gains : numpy.ndarray of shape (n_points,)
        The Y-axis values representing the cumulative gain, i.e., the proportion of true positives 
        captured as a function of the total instances considered. Values range from 0 (no positives 
        captured) to 1 (all positives captured), and an initial 0 is inserted at the start to represent 
        the baseline.

    Raises
    ------
    ValueError
        - If `y_true` does not contain exactly two distinct classes, indicating that the problem is 
          not a binary classification task.
        - If `pos_label` is not provided and cannot be inferred from `y_true`, or if `y_score` is binary 
          instead of continuous.
        - If the positive class does not appear in `y_true`, resulting in a gain of zero, which would 
          mislead the user.

    Notes
    -----
    - **Binary Classification Only:** This implementation is strictly for binary classification. 
      Multi-class problems are not supported and will result in a `ValueError`.
    - **Score Type:** The `y_score` array must contain continuous values. Binary scores (0/1) are 
      not appropriate for plotting cumulative gain curves and will lead to incorrect results.
    - **Performance:** The function sorts the scores, which contributes to a time complexity of 
      O(n log n), where `n` is the number of samples. For large datasets, this could be a 
      performance bottleneck.
    - **Baseline Insertion:** A starting point of (0, 0) is included in both the `percentages` and `gains` 
      arrays. This ensures that the cumulative gain curve starts at the origin, providing an accurate 
      representation of the gain from zero instances considered.
    - **Handling Edge Cases:** If `y_true` contains no instances of the positive class, the function 
      will raise a `ValueError`, as a cumulative gain curve would not be meaningful.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> import matplotlib.pyplot as plt
    >>> # Generate a binary classification dataset
    >>> X, y = make_classification(n_samples=1000, n_classes=2, n_informative=3, random_state=42)
    >>> # Split into training and test sets
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    >>> # Train a logistic regression model
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> # Predict probabilities for the test set
    >>> y_scores = model.predict_proba(X_test)[:, 1]
    >>> # Calculate the cumulative gain curve
    >>> percentages, gains = cumulative_gain_curve(y_test, y_scores)
    >>> # Plot the cumulative gain curve
    >>> plt.plot(percentages, gains, marker='o')
    >>> plt.xlabel('Percentage of Samples')
    >>> plt.ylabel('Gain')
    >>> plt.title('Cumulative Gain Curve')
    >>> plt.grid()
    >>> plt.show()
    """
    # Convert input to numpy arrays for efficient processing
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Identify the unique classes in y_true and ensure it is binary
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            "`y_true` must contain exactly two distinct classes "
            "for binary classification."
        )

    # Ensure the pos_label is provided or infer it from the data
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError(
            "Data is not binary and pos_label is not specified. "
            "Please provide the `pos_label`."
        )
    elif pos_label is None:
        pos_label = 1.  # Default to 1 for standard binary labels

    # Convert y_true to a boolean array where the positive class is True
    y_true = (y_true == pos_label)

    # Ensure y_score is continuous and not binary
    if (np.array_equal(np.unique(y_score), [0, 1]) or
        np.array_equal(np.unique(y_score), [-1, 1]) or
        np.array_equal(np.unique(y_score), [0]) or
        np.array_equal(np.unique(y_score), [-1]) or
        np.array_equal(np.unique(y_score), [1])
       ):
        raise ValueError(
            "`y_score` should contain continuous values, "
            "not binary (0/1) scores. Provide non-thresholded scores."
        )

    # Sort instances by their scores in descending order
    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    y_score = y_score[sorted_indices]

    # total number of positive instances
    total_positives = float(np.sum(y_true))
    if total_positives == 0:
        raise ValueError(
            "The positive class does not appear in `y_true`, "
            "resulting in a gain of zero."
        )

    # Compute cumulative gains (number of true positives as threshold decreases)
    gains = np.cumsum(y_true)
    # Normalize gains by the total number of positive instances
    gains = gains / float(total_positives)

    # Calculate the cumulative percentage of instances considered
    percentages = np.arange(start=1, stop=len(y_true) + 1)
    # Normalize percentages by the total number of instances
    percentages = percentages / float(len(y_true))

    # Insert (0, 0) as the starting point for baseline
    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains


def binary_ks_curve(
    y_true, 
    y_probas
):
    """
    Generate the data points necessary to plot the Kolmogorov-Smirnov (KS)
    curve for binary classification tasks.

    The KS Statistic measures the maximum vertical distance between
    the cumulative distribution functions (CDFs) of the predicted
    probabilities for the positive and negative classes.
    It is used to evaluate the discriminatory power of a binary classifier.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels of the data. This array should contain exactly
        two unique classes representing a binary classification problem.
        If more than two classes are present, the function will raise a 
        `ValueError`.

    y_probas : array-like of shape (n_samples,)
        Probability predictions for the positive class.
        This array should contain continuous values representing 
        the predicted probability of the positive class.

    Returns
    -------
    thresholds : numpy.ndarray of shape (n_thresholds,)
        An array containing the threshold (X-axis) values
        used for plotting the KS curve. These thresholds
        range from the minimum to the maximum predicted probabilities.

    pct1 : numpy.ndarray of shape (n_thresholds,)
        An array containing the cumulative (Y-axis) percentage of samples
        for the positive class up to each threshold value.

    pct2 : numpy.ndarray of shape (n_thresholds,)
        An array containing the cumulative (Y-axis) percentage of samples
        for the negative class up to each threshold value.

    ks_statistic : float
        The KS Statistic, which is the maximum vertical distance
        between the cumulative distribution functions 
        of the positive and negative classes.

    max_distance_at : float
        The threshold (X-axis) value at which the maximum vertical distance
        between the two cumulative distribution functions 
        (and hence the KS Statistic) is observed.

    classes : numpy.ndarray of shape (2,)
        An array containing the labels of the two classes present in ``y_true``.

    Raises
    ------
    ValueError
        - If ``y_true`` does not contain exactly two distinct classes,
          indicating that the problem is not binary.
        - If ``y_probas`` contains binary values instead of continuous probabilities.

    Notes
    -----
    - **Binary Classification Only:** This implementation is strictly for binary classification. 
      Multi-class problems are not supported and will result in a `ValueError`.
    - **Probability Scores:** The `y_probas` array must contain continuous values representing probabilities.
      Binary scores (0/1) are not appropriate for KS curve calculations.
    - **Performance:** The function sorts the predicted probabilities for both classes, leading to a time 
      complexity of O(n log n) where `n` is the number of samples. Sorting could be a performance bottleneck 
      for very large datasets.
    - **Handling Edge Cases:** The function inserts thresholds of 0 and 1 if they are not already present to ensure 
      that the KS curve starts and ends at the boundaries of the predicted probability range.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Kolmogorov-Smirnov (KS) Statistic
    
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> import matplotlib.pyplot as plt
        >>> from scikitplot.utils._helpers import binary_ks_curve
        >>> # Generate a binary classification dataset
        >>> X, y = make_classification(n_samples=1000, n_classes=2, n_informative=3, random_state=0)
        >>> # Split into training and test sets
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> # Train a logistic regression model
        >>> model = LogisticRegression()
        >>> model.fit(X_train, y_train)
        >>> # Predict probabilities for the test set
        >>> y_probas = model.predict_proba(X_test)[:, 1]
        >>> # Calculate the KS Statistic curve
        >>> thresholds, pct1, pct2, ks_statistic, max_distance_at, classes = binary_ks_curve(y_test, y_probas)
        >>> # Plot the KS Statistic curve
        >>> plt.plot(thresholds, pct1 - pct2, marker='o')
        >>> plt.xlabel('Threshold')
        >>> plt.ylabel('KS Statistic')
        >>> plt.title('KS Statistic Curve')
        >>> plt.grid()
        >>> plt.show()
    """
    # Convert input to numpy arrays for efficient processing
    y_true = np.asarray(y_true)
    y_probas = np.asarray(y_probas)

    # Encode the true labels to ensure binary classification
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError(
            f'Cannot calculate KS statistic for data with '
            f'{len(lb.classes_)} category/ies.'
        )

    # Separate probabilities for the two classes
    # neg_prob = y_probas[y_true == 0]
    # pos_prob = y_probas[y_true == 1]
    idx = encoded_labels == 0
    
    # Sort the predicted probabilities for both classes
    data1 = np.sort(y_probas[idx])  # Probabilities for the negative class
    data2 = np.sort(y_probas[~idx])  # Probabilities for the positive class
    # data2 = np.sort(y_probas[np.logical_not(idx)])

    # Initialize counters and lists to store results
    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []

    # Compute cumulative percentages for different thresholds
    while ctr1 < len(data1) or ctr2 < len(data2):

        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1

        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1

        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)

    # Convert lists to numpy arrays
    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    # Insert boundary values if not present
    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])

    # Calculate the KS Statistic and the threshold where it occurs
    differences = pct1 - pct2
    ks_statistic, max_distance_at = (
        np.max(differences),
        thresholds[np.argmax(differences)]
    )
    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_