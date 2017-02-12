from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sklearn.preprocessing import LabelEncoder


def binary_ks_curve(y_true, y_probas):
    """This function generates the points necessary to calculate the KS Statistic curve.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_probas (array-like, shape (n_samples)): Probability predictions of the positive class.

    Returns:
        thresholds (numpy.ndarray): An array containing the X-axis values for plotting the
            KS Statistic plot.

        pct1 (numpy.ndarray): An array containing the Y-axis values for one curve of the
            KS Statistic plot.

        pct2 (numpy.ndarray): An array containing the Y-axis values for one curve of the
            KS Statistic plot.

        ks_statistic (float): The KS Statistic, or the maximum vertical distance between the
            two curves.

        max_distance_at (float): The X-axis value at which the maximum vertical distance between
            the two curves is seen.

        classes (np.ndarray, shape (2)): An array containing the labels of the two classes making
            up `y_true`.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The KS Statistic is only relevant in
            binary classification.
    """
    y_true, y_probas = np.asarray(y_true), np.asarray(y_probas)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError('Cannot calculate KS statistic for data with '
                         '{} category/ies'.format(len(lb.classes_)))
    idx = encoded_labels == 0
    data1 = np.sort(y_probas[idx])
    data2 = np.sort(y_probas[-idx])

    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
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

    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])

    differences = pct1 - pct2
    ks_statistic, max_distance_at = np.max(differences), thresholds[np.argmax(differences)]

    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_
