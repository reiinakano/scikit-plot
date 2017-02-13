# Welcome to Scikit-plot

[![PyPI version](https://badge.fury.io/py/scikit-plot.svg)](https://badge.fury.io/py/scikit-plot)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()

## Scikit-plot? As in Scikit-learn with plotting?

Yes. Scikit-plot is the result of an unartistic data scientist's dreadful realization that *visualization is one of the most crucial components in the data science process, not just a mere afterthought*.

Gaining insights is simply a lot easier when you're looking at a colored heatmap of a confusion matrix complete with class labels rather than a single-line dump of numbers enclosed in brackets. Besides, if you ever need to present your results to someone (virtually any time anybody hires you to do data science), you show them visualizations, not a bunch of numbers in Excel.

That said, there are a number of visualizations that frequently pop up in machine learning. Scikit-plot is a humble attempt to provide aesthetically-challenged programmers (such as myself) the opportunity to generate quick and beautiful graphs and plots with as little boilerplate as possible.

## Okay then, prove it. Show us an example.

Say we use Naive Bayes in multi-class classification and decide we want to visualize the results of a common classification metric, the Area under the Receiver Operating Characteristic curve. Since the ROC is only valid in binary classification, we want to show the respective ROC of each class if it were the positive class. As an added bonus, let's show the micro-averaged and macro-averaged curve in the plot as well.

Using scikit-plot with the sample digits dataset from scikit-learn.

```python
from sklearn.datasets import load_digits as load_data
from sklearn.naive_bayes import GaussianNB

# This is all that's needed for scikit-plot
import matplotlib.pyplot as plt
from scikitplot import classifier_factory

X, y = load_data(return_X_y=True)
nb = GaussianNB()
classifier_factory(nb)
nb.plot_roc_curve(X, y, random_state=1)
plt.show()
```
![roc_curves](examples/roc_curves.png)

Pretty.

So what happened here? First, we created a regular Naive Bayes classifier instance from scikit-learn and assigned it to `nb`. We then passed `nb` to `classifier_factory`. Then, like magic, we call `nb`'s *instance method* `plot_roc_curve` and pass it a features array and corresponding label array. Finally, we call `plt.show()` to display the corresponding plot.

Wait, what? The scikit-learn `GaussianNB` classifier doesn't have a `plot_roc_curve` method. How does this not throw an error? Well, `classifier_factory` is a function that modifies an __instance__ of a scikit-learn classifier. When we passed `nb` to `classifier_factory`, it __appended__ new plotting methods to the instance, one of which was `plot_roc_curve`, while leaving everything else alone.

This means that our classifier instance `nb` will behave the same way as before, with all its original variables and methods intact. In fact, if you take any of your existing scripts, pass your classifier instances to `classifier_factory` at the top and run them, you'll likely never notice a difference!

`classifier_factory` adds a lot more plotting methods to classifier instances. Visit the [docs](http://scikit-plot.readthedocs.io/en/latest/) for a complete list of what you can accomplish.

Finally, compare and [view the non-scikit-plot way of plotting the multi-class ROC curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html). Which one would you rather do?

## Installation

Installation is simple! First, make sure you have the dependencies [Scikit-learn](http://scikit-learn.org) and [Matplotlib](http://matplotlib.org/) installed.

Then just run:
```bash
pip install scikit-plot
```

Or if you want, clone this repo and run
```bash
python setup.py install
```
at the root folder.

## Documentation and Examples

Explore the full features of Scikit-plot.

You can find detailed documentation [here](http://scikit-plot.readthedocs.io/en/latest/).

Examples are found in the [examples folder of this repo](examples/).

Happy plotting!
