"""
An example showing the plot_roc_curve method
used by a scikit-learn classifier
"""
from __future__ import absolute_import
from sklearn.datasets import load_digits as load_data
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.svm import SVC

X, y = load_data(return_X_y=True)
X, y = shuffle(X, y)
clf = SVC()
param_name = 'C'
param_range = [.1,10,100]
skplt.estimators.plot_validation_curve(clf, X, y, param_name=param_name, param_range=param_range)
plt.show()
