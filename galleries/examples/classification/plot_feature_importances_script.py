"""
plot_feature_importances with examples
======================================

An example showing the :py:func:`~scikitplot.api.estimators.plot_feature_importances` function
used by a scikit-learn classifier.
"""

# Authors: scikit-plots developers
# License: MIT

from sklearn.datasets import (
    make_classification,
    load_breast_cancer as data_2_classes,
    load_iris as data_3_classes,
    load_digits as data_10_classes,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np; np.random.seed(0)
# importing pylab or pyplot
import matplotlib.pyplot as plt

# Import scikit-plot
# import scikitplot as skplt
import scikitplot.api as skplt

# Load the data
X, y = data_3_classes(return_X_y=True, as_frame=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)

# Create an instance of the LogisticRegression
model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

# Plot!
ax, features = skplt.estimators.plot_feature_importances(
    model, 
    feature_names=['petal length', 'petal width', 'sepal length', 'sepal width'],
);

# Adjust layout to make sure everything fits
plt.tight_layout()
# Save the plot to a file
# plt.savefig('plot_feature_importances_script.png')
# Display the plot
plt.show(block=True)

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#
# .. tags::
#
#    model-type: Classification
#    model-workflow: Model Evaluation
#    plot-type: bar
#    level: beginner
#    purpose: showcase