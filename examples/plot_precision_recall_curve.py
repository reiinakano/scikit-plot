"""An example showing the plot_precision_recall method used by a scikit-learn classifier"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data


X, y = load_data(return_X_y=True)
nb = classifier_factory(GaussianNB())
nb.plot_precision_recall_curve(X, y, random_state=1)
plt.show()

# Using the more flexible functions API
from scikitplot import plotters
nb = GaussianNB()
nb = nb.fit(X, y)
probas = nb.predict_proba(X)
plotters.plot_precision_recall_curve(y_true=y, y_probas=probas)
plt.show()
