"""
An example showing the plot_feature_importances
method used by a scikit-learn classifier
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris as load_data
import matplotlib.pyplot as plt
import scikitplot as skplt

X, y = load_data(return_X_y=True)
rf = RandomForestClassifier()
rf.fit(X, y)
skplt.estimators.plot_feature_importances(rf,
                                          feature_names=['petal length',
                                                         'petal width',
                                                         'sepal length',
                                                         'sepal width'])
plt.show()
