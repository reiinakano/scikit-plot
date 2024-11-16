"""
plot_residuals_distribution with examples
==========================================

An example showing the :py:func:`~scikitplot.api.metrics.plot_residuals_distribution` function
used by a scikit-learn regressor.
"""

# Authors: scikit-plots developers
# License: BSD-3 Clause

from sklearn.datasets import (
    make_classification,
    load_diabetes as load_data,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import numpy as np; np.random.seed(0)
# importing pylab or pyplot
import matplotlib.pyplot as plt

# Import scikit-plot
# import scikitplot as skplt
import scikitplot.api as skplt
import scikitplot.probscale as probscale

# Load the data
X, y = load_data(return_X_y=True, as_frame=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)

# Create an instance of the LogisticRegression
model = LinearRegression().fit(X_train, y_train)

# Perform predictions
y_val_pred = model.predict(X_val)

# Plot!
ax = skplt.metrics.plot_residuals_distribution(
    y_val, y_val_pred, dist_type='normal'
);
# Adjust layout to make sure everything fits
plt.tight_layout()
# Save the plot to a file
# plt.savefig('plot_residuals_distribution_script.png')
# Display the plot
plt.show(block=True)

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - https://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
#    - https://online.stat.psu.edu/stat462/node/122/
#
# .. tags::
#
#    model-type: Regression
#    model-workflow: Model Evaluation
#    plot-type: histogram
#    plot-type: qqplot
#    domain: statistics
#    level: intermediate
#    purpose: showcase