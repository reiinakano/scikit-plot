"""An example showing the plot_silhouette method used by a scikit-learn clusterer"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from scikitplot import clustering_factory
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as load_data


X, y = load_data(return_X_y=True)
kmeans = clustering_factory(KMeans(random_state=1))
kmeans.plot_elbow_curve(X, cluster_ranges=range(1, 11))
plt.show()
