"""An example showing the plot_pca_component_variance method used by a scikit-learn PCA object"""
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits as load_data
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt


X, y = load_data(return_X_y=True)
pca = PCA(random_state=1)
pca.fit(X)
skplt.plot_pca_component_variance(pca)
plt.show()
