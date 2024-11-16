.. _installation_instructions_conda:


.. warning::

    Not Implemented...

Install conda using the
`miniforge installers <https://github.com/conda-forge/miniforge#miniforge>`__ (no
administrator permission required). Then run:

.. prompt:: bash >>> auto

  conda create -n sklearn-env -c conda-forge scikit-plots
  conda activate sklearn-env

In order to check your installation, you can use:

.. prompt:: bash >>> auto

  conda list scikit-plots  # show scikit-plots version and location
  conda list               # show all installed packages in the environment
  python -c "import scikitplot; scikitplot.show_versions()"