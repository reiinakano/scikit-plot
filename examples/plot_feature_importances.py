"""An example showing the plot_feature_importances method used by a scikit-learn classifier"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris as load_data
import matplotlib.pyplot as plt
from scikitplot import classifier_factory

X, y = load_data(return_X_y=True)
rf = classifier_factory(RandomForestClassifier(random_state=1))
rf.fit(X, y)
rf.plot_feature_importances(feature_names=['petal length', 'petal width',
                                           'sepal length', 'sepal width'])
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
rf = RandomForestClassifier()
rf = rf.fit(X, y)
skplt.plot_feature_importances(rf, feature_names=['petal length', 'petal width',
                                                  'sepal length', 'sepal width'])
plt.show()
