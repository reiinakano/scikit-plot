# Welcome to Scikit-plot

[![PyPI version](https://badge.fury.io/py/scikit-plot.svg)](https://badge.fury.io/py/scikit-plot)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()

## Scikit-plot? As in Scikit-learn with plotting?

Yes. Scikit-plot is the result of an unartistic data scientist's dreadful realization that *visualization is not just a mere afterthought, but one of the most crucial components in the data science process*.

Getting insights from ML algorithms is simply a lot more pleasant when you're looking at a pretty heatmap of a confusion matrix complete with class labels rather than a single-line dump of numbers enclosed in brackets. Also, if you ever need to present your results (virtually any time somebody hires you to do data science), you show visualizations, not a bunch of numbers in Excel.

That said, there are a number of visualizations that frequently pop up in machine learning. Scikit-plot is a humble attempt to provide aesthetically-challenged programmers (such as myself) the opportunity to generate quick and beautiful graphs and plots with as little boilerplate as possible.

## Okay then, prove it. Show us an example.

Say we use Naive Bayes in multi-class classification and decide we want to visualize the results of a common classification metric, the Area under the Receiver Operating Characteristic curve. Since the ROC is only valid in binary classification, we want to show the respective ROC of each class if it were the positive class. As an added bonus, let's show the micro-averaged and macro-averaged curve in the plot as well.

Using scikit-plot with the sample digits dataset from scikit-learn.

```python
from sklearn.datasets import load_digits as load_data
from sklearn.naive_bayes import GaussianNB

# This is all that's needed for scikit-plot
import matplotlib.pyplot as plt
from scikitplot.scikitplot import classifier_factory

X, y = load_data(return_X_y=True)
nb = GaussianNB()
classifier_factory(nb)
nb.plot_roc_curve(X, y, random_state=1)
plt.show()
```
![roc_curves](examples/roc_curves.png)

So what happened here? First, `classifier_factory` is a function that modifies an __instance__ of a scikit-learn classifier. The `classifier_factory` function merely __appends__ new plotting methods to the instance, one of which is `plot_roc_curve`, while leaving everything else alone. 

This means that the instance will behave the same way as before, with all its original variables and methods intact. In fact, if you pass your classifier to `classifier_factory` at the top of any of your existing scripts and run them, you'll likely never notice a difference!

`classifier_factory` adds a lot more plotting methods to classifier instances and they are all as easy to use as that first example. Visit the docs for a complete list of what you can accomplish.
