"""
An example showing the plot_silhouette method
used by a scikit-learn clusterer
"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as load_data
import scikitplot as skplt

X, y = load_data(return_X_y=True)
kmeans = KMeans(n_clusters=4, random_state=1)
cluster_labels = kmeans.fit_predict(X)
skplt.metrics.plot_silhouette(X, cluster_labels)
plt.show()
