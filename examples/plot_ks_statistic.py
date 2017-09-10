"""
An example showing the plot_ks_statistic method used
by a scikit-learn classifier
"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer as load_data
import scikitplot as skplt


X, y = load_data(return_X_y=True)
lr = LogisticRegression()
lr.fit(X, y)
probas = lr.predict_proba(X)
skplt.metrics.plot_ks_statistic(y_true=y, y_probas=probas)
plt.show()
