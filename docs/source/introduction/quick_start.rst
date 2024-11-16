.. _quick_start:

.. title:: scikit-plots: Machine Learning Visualization with Python

===========
Quick Start
===========

This guide provides a quick introduction to plotting with scikit-plots.

1. **Install Scikit-plots**:

- Use pip to install Scikit-plots::

    >>> pip install scikit-plots


A Simple Example
----------------

Let's start with a basic example where we use a Random Forest classifier to evaluate the digits dataset provided by Scikit-learn.

A common way to assess a classifier's performance is through its confusion matrix. Here’s how we can do it:

1. **Load the Dataset**: 
   We'll use the digits dataset, which contains features and labels for classification.
   
2. **Initialize the Classifier**: 
   Create a :class:`~sklearn.ensemble.RandomForestClassifier` with specified parameters.
   
3. **Generate Predictions**: 
   Use :func:`~sklearn.model_selection.cross_val_predict` to obtain predicted labels through cross-validation. This function provides cross-validated estimates for each sample point, which helps in evaluating metrics like accuracy, precision, recall, and the confusion matrix.

4. **Plot the Confusion Matrix**: 
   Use :py:func:`~scikitplot.api.metrics.plot_classifier_eval` to visualize the confusion matrix.

5. **Display the Plot**: 
   Optionally, use :py:func:`~matplotlib.pyplot.show` to display the plot.

Here’s the code to illustrate the process:

.. plot::
    :context: close-figs
    :align: center
    
    >>> from sklearn.datasets import load_digits
    >>> X, y = load_digits(return_X_y=True)
    >>> 
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=1)
    >>> 
    >>> from sklearn.model_selection import cross_val_predict
    >>> y_pred = cross_val_predict(clf, X, y)
    >>> 
    >>> import scikitplot as skplt
    >>> fig1 = skplt.metrics.plot_classifier_eval(
    >>>     y, y_pred,
    >>>     labels=np.unique(y),
    >>>     figsize=(8, 3.2),
    >>>     title='Confusion Matrix'
    >>> );

The resulting confusion matrix shows how well the classifier performs. In this case, it struggles with digits 1, 8, and 9. Fine-tuning the Random Forest’s hyperparameters might improve performance.


One More Example
----------------

**Maximum flexibility. Compatibility with non-scikit-learn objects.**

Although Scikit-plot is loosely based around the scikit-learn interface, you don't actually need Scikit-learn objects to use the available functions. As long as you provide the functions what they're asking for, they'll happily draw the plots for you.

Use `Tensorflow <https://www.tensorflow.org>`_ or `Pytorch <https://pytorch.org>`_

Here's a quick example to generate the precision-recall curves of a :py:mod:`~torch` or :py:mod:`~tensorflow.keras` Keras classifier model on a sample dataset.

.. plot::
    :context: close-figs
    :align: center
    
    >>> import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.model_selection import train_test_split
    >>> import matplotlib.pyplot as plt
    >>> import scikitplot as skplt
    >>> 
    >>> # Load the digits dataset
    >>> X, y = load_digits(return_X_y=True)
    >>> 
    >>> # Split the dataset into training and validation sets
    >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
    >>> 
    >>> # Convert labels to one-hot encoding
    >>> Y_train = tf.keras.utils.to_categorical(y_train)
    >>> Y_val = tf.keras.utils.to_categorical(y_val)
    >>> 
    >>> # Define a simple TensorFlow model
    >>> model = tf.keras.Sequential([
    >>>     tf.keras.layers.Input(shape=(X_train.shape[1],)),
    >>>     tf.keras.layers.Dense(64, activation='relu'),
    >>>     tf.keras.layers.Dense(32, activation='relu'),
    >>>     tf.keras.layers.Dense(10, activation='softmax')
    >>> ])
    >>> 
    >>> # Compile the model
    >>> model.compile(optimizer='adam',
    >>>               loss='categorical_crossentropy',
    >>>               metrics=['accuracy'])
    >>> 
    >>> # Train the model
    >>> model.fit(
    >>>     X_train, Y_train,
    >>>     batch_size=64,
    >>>     epochs=10,
    >>>     validation_data=(X_val, Y_val),
    >>>     verbose=0
    >>> )
    >>> 
    >>> # Predict probabilities on the validation set
    >>> y_probas = model.predict(X_val)
    >>> 
    >>> # Plot precision-recall curves
    >>> skplt.metrics.plot_precision_recall(y_val, y_probas)
    >>> plt.show()


Just pass the ground truth labels and predicted probabilities to
:py:func:`~scikitplot.api.metrics.plot_precision_recall` to generate the precision-recall curves.
This method is flexible and works with any classifier that produces predicted probabilities,
from Keras classifiers to NLTK Naive Bayes to XGBoost as long as you pass in the predicted probabilities
in the correct format.


Now what?
---------

The recommended way to start using Scikit-plot is to just go through the documentation for the various modules and choose which plots you think would be useful for your work.

Happy plotting!