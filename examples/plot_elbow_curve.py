"""
An example showing the plot_silhouette
method used by a scikit-learn clusterer
"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as load_data


X, y = load_data(return_X_y=True)
kmeans = KMeans(random_state=1)
skplt.cluster.plot_elbow_curve(kmeans, X, cluster_ranges=range(1, 11))
plt.show()
