# Welcome to 101 Scikit-plots

## Single line functions for detailed visualizations

The quickest and easiest way to go from analysis...

## Sample Plots

<table style="margin-left:auto;margin-right:auto;width:100%;border-collapse:collapse;">
  <tr>
    <th style="width:50%;text-align:center;">Sample Plot 1</th>
    <th style="width:50%;text-align:center;">Sample Plot 2</th>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_learning_curve.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-estimators-plot_learning_curve-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_calibration_curve.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-metrics-plot_calibration_curve-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_classifier_eval.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-metrics-plot_classifier_eval-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_feature_importances.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-estimators-plot_feature_importances-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_roc.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-metrics-plot_roc-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_precision_recall.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-metrics-plot_precision_recall-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_pca_component_variance.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-decomposition-plot_pca_component_variance-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_pca_2d_projection.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-decomposition-plot_pca_2d_projection-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_elbow.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-cluster-plot_elbow-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_silhouette.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-metrics-plot_silhouette-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_cumulative_gain.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-deciles-plot_cumulative_gain-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_lift.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/muhammed-dev/dev/_images/scikitplot-deciles-plot_lift-1.png">
    </td>
  </tr>
</table>


Scikit-plots is the result of an unartistic data scientist's dreadful realization that *visualization is one of the most crucial components in the data science process, not just a mere afterthought*.

Gaining insights is simply a lot easier when you're looking at a colored heatmap of a confusion matrix complete with class labels rather than a single-line dump of numbers enclosed in brackets. Besides, if you ever need to present your results to someone (virtually any time anybody hires you to do data science), you show them visualizations, not a bunch of numbers in Excel.

That said, there are a number of visualizations that frequently pop up in machine learning. Scikit-plots is a humble attempt to provide aesthetically-challenged programmers (such as myself) the opportunity to generate quick and beautiful graphs and plots with as little boilerplate as possible.

## Okay then, prove it. Show us an example.

Say we use [Keras Classifier](https://keras.io/api/models/sequential/) in multi-class classification and decide we want to visualize the results of a common classification metric, such as sklearn's [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) with a [confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html).

Let’s start with a basic example where we use a Keras classifier to evaluate the digits dataset provided by Scikit-learn.

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scikitplot as skplt

# Load the digits dataset
X, y = load_digits(return_X_y=True)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

# Convert labels to one-hot encoding
Y_train = tf.keras.utils.to_categorical(y_train)
Y_val = tf.keras.utils.to_categorical(y_val)

# Define a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    X_train, Y_train,
    batch_size=64,
    epochs=10,
    validation_data=(X_val, Y_val),
    verbose=0
)

# Predict probabilities on the validation set
y_probas = model.predict(X_val)

# Plot precision-recall curves
skplt.metrics.plot_precision_recall(y_val, y_probas)
plt.show()
```

<div align=center>
  <img style="display:block;width:60%;height:auto;align:center;" alt="quick_start" src="https://scikit-plots.github.io/stable/_images/quick_start-2.png">
</div>

Pretty.

## Maximum flexibility. Compatibility with non-scikit-learn objects.

Although Scikit-plot is loosely based around the scikit-learn interface, you don't actually need scikit-learn objects to use the available functions. 
As long as you provide the functions what they're asking for, they'll happily draw the plots for you.

The possibilities are endless.

## User Installation

1. **Install Scikit-plots**:
   - Use pip to install Scikit-plots:

     ```bash
     pip install scikit-plots
     ```

## Release Notes

See the [changelog](https://scikit-plots.github.io/stable/whats_new/whats_new.html)
for a history of notable changes to scikit-plots.

## Documentation and Examples

Explore the full features of Scikit-plot.

## Contributing to scikit-plots

Reporting a bug? Suggesting a feature? Want to add your own plot to the library? Visit our.

## Citing scikit-plots

1. scikit-plots, “scikit-plots: vlatest”. Zenodo, Aug. 23, 2024.
   DOI: [10.5281/zenodo.13367000](https://doi.org/10.5281/zenodo.13367000).

2. scikit-plots, “scikit-plots: v0.3.8dev0”. Zenodo, Aug. 23, 2024.
   DOI: [10.5281/zenodo.13367001](https://doi.org/10.5281/zenodo.13367001).