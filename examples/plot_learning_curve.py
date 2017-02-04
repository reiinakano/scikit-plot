"""An example showing the plot_learning_curve method used by a scikit-learn classifier"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot.scikitplot import classifier_factory
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer


X, y = load_breast_cancer(return_X_y=True)
rf = RandomForestClassifier()
rf = classifier_factory(rf)
rf.plot_learning_curve(X, y)
plt.show()
