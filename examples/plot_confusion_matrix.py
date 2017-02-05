from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits as load_data
import matplotlib.pyplot as plt
from scikitplot.scikitplot import classifier_factory

X, y = load_data(return_X_y=True)
rf = classifier_factory(RandomForestClassifier())
rf.plot_confusion_matrix(X, y, normalize=True)
plt.show()
