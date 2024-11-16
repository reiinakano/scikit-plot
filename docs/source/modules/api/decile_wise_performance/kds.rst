.. currentmodule:: scikitplot.api.kds

.. _kds:

KeyToDataScience
======================================================================

This module contains functions related to :py:mod:`~scikitplot.api.kds`.

.. seealso::

   [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py


kds Plots
----------------------------------------------------------------------

* Let's introduce the Gains, Lift and (cumulative) Response plots.

.. _plot_cumulative_gain:

plot cumulative gain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.kds.plot_cumulative_gain`

The mathematical formulation is the following:

.. math::

    \hat{K}

Trained model of :class:`~sklearn.linear_model.LogisticRegression` and
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_kds_plot_cumulative_gain_script.py`: Example usage of
  :class:`~sklearn.linear_model.LogisticRegression` using the iris dataset

* :ref:`sphx_glr_auto_examples_kds_plot_cumulative_gain_script.py`: Example usage of
  :class:`~sklearn.ensemble.RandomForestClassifier` using the iris dataset


.. dropdown:: References

  * cumulative_gain



.. _plot_lift:

plot lift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.kds.plot_lift`

.. note:: **Lift Curve**

Trained model of :class:`~sklearn.linear_model.LogisticRegression` and
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_kds_plot_lift_script.py`: Example usage of
  :class:`~sklearn.linear_model.LogisticRegression` using the iris dataset

* :ref:`sphx_glr_auto_examples_kds_plot_lift_script.py`: Example usage of
  :class:`~sklearn.ensemble.RandomForestClassifier` using the iris dataset


.. dropdown:: References

  * lift_curve



.. _plot_ks_statistic:

plot_ks_statistic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.kds.plot_ks_statistic`

Trained model of :class:`~sklearn.linear_model.LogisticRegression` and
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_kds_plot_ks_statistic_script.py`: Example usage of
  :class:`~sklearn.linear_model.LogisticRegression` using the iris dataset

* :ref:`sphx_glr_auto_examples_kds_plot_ks_statistic_script.py`: Example usage of
  :class:`~sklearn.ensemble.RandomForestClassifier` using the iris dataset


.. dropdown:: References

  * ks_statistic



.. _report:

report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~scikitplot.api.kds.report`

Trained model of :class:`~sklearn.linear_model.LogisticRegression` and
:class:`~sklearn.ensemble.RandomForestClassifier`. For an example of
performing image:

.. rubric:: Examples

* report


.. dropdown:: References

  * report