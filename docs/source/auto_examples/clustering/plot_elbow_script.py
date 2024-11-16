"""
plot_elbow with examples
=============================

An example showing the :py:func:`~scikitplot.api.estimators.plot_elbow` function
used by a scikit-learn clusterer.
"""

# Authors: scikit-plots developers
# License: MIT

# %%
# Load the dataset
# ----------------
#
# We will start by loading the ``iris`` dataset.
from sklearn.datasets import (
    make_classification,
    load_breast_cancer as data_2_classes,
    load_iris as data_3_classes,
    load_digits as data_10_classes,
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
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

# %%
# Model Training
# --------------
#
# Create an instance of the LogisticRegression
model = KMeans(n_clusters=4, random_state=1)


# %%
# Visualize the results
# ---------------------
#
# Plot!
ax = skplt.estimators.plot_elbow(
    model, 
    X_train, 
    cluster_ranges=range(1, 11)
);

# Adjust layout to make sure everything fits
plt.tight_layout()
# Save the plot to a file
# plt.savefig('plot_elbow_script.png')
# Display the plot
plt.show(block=True)

# %%
#
# .. tags::
#
#    model-type: Clustering
#    model-type: K-Means
#    model-workflow: Model Evaluation
#    plot-type: line
#    plot-type: WSS (within-cluster sum of squares)
#    plot-type: inertia (sum of squared distances)
#    level: beginner
#    purpose: showcase