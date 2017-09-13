"""
An example showing the plot_calibration_curve method
used by a scikit-learn classifier
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import scikitplot as skplt

X, y = make_classification(n_samples=100000, n_features=20,
                           n_informative=2, n_redundant=2,
                           random_state=20)

X_train, y_train, X_test, y_test = X[:1000], y[:1000], X[1000:], y[1000:]

rf_probas = RandomForestClassifier().fit(X_train, y_train).predict_proba(X_test)
lr_probas = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)
nb_probas = GaussianNB().fit(X_train, y_train).predict_proba(X_test)
sv_scores = LinearSVC().fit(X_train, y_train).decision_function(X_test)

probas_list = [rf_probas, lr_probas, nb_probas, sv_scores]
clf_names=['Random Forest',
           'Logistic Regression',
           'Gaussian Naive Bayes',
           'Support Vector Machine']

skplt.metrics.plot_calibration_curve(y_test,
                                     probas_list=probas_list,
                                     clf_names=clf_names,
                                     n_bins=10)
plt.show()
