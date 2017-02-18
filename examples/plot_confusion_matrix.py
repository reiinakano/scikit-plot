"""An example showing the plot_confusion_matrix method used by a scikit-learn classifier"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits as load_data
import matplotlib.pyplot as plt
from scikitplot import classifier_factory

X, y = load_data(return_X_y=True)
rf = classifier_factory(RandomForestClassifier())
rf.plot_confusion_matrix(X, y, normalize=True)
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
rf = RandomForestClassifier()
rf = rf.fit(X, y)
preds = rf.predict(X)
skplt.plot_confusion_matrix(y_true=y, y_pred=preds)
plt.show()
