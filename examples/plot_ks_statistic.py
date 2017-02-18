"""An example showing the plot_ks_statistic method used by a scikit-learn classifier"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer as load_data


X, y = load_data(return_X_y=True)
lr = classifier_factory(LogisticRegression())
lr.plot_ks_statistic(X, y, random_state=1)
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
lr = LogisticRegression()
lr = lr.fit(X, y)
probas = lr.predict_proba(X)
skplt.plot_ks_statistic(y_true=y, y_probas=probas)
plt.show()
