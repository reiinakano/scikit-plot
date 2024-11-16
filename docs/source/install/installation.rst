.. _installing:

=======================
Installing scikit-plots
=======================

Scikit-plots relies on `Matplotlib <http://matplotlib.org/>`_ and
`Scikit-learn <http://scikit-learn.org/>`_ for plotting model results.
Additionally, it can work with results from other modeling libraries such as
`TensorFlow <https://www.tensorflow.org/>`_ and `PyTorch <https://pytorch.org/>`_.
Ensure that you have the necessary libraries installed in your environment.

There are different ways to install scikit-plots:

* :ref:`Install the latest official release <install_official_release>`. This
  is the best approach for most users. It will provide a stable version
  and pre-built packages are available for most platforms.

.. Note
    * :ref:`Building the package from source
      <install_bleeding_edge>`. This is best for users who want the
      latest-and-greatest features and aren't afraid of running
      brand-new code. This is also needed for users who wish to contribute to the
      project.


.. _install_official_release:

Installing the latest release
=============================

.. raw:: html

  <style>
    /* Show caption on large screens */
    @media screen and (min-width: 960px) {
      .install-instructions .sd-tab-set {
        --tab-caption-width: 20%;
      }

      .install-instructions .sd-tab-set.tabs-package-manager::before {
        content: "Package Manager";
      }

      .install-instructions .sd-tab-set.tabs-os::before {
        content: "Operating System";
      }
    }
  </style>

.. div:: install-instructions

  .. tab-set::
    :class: tabs-package-manager

    .. tab-item:: pip
      :class-label: tab-6

      .. tab-set::
        :class: tabs-os

        .. tab-item:: Windows
          :class-label: tab-4
          :sync: package-manager-pip

          Install the 64-bit version of Python 3, for instance from the
          `official website <https://www.python.org/downloads/windows/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install scikit-plots.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: powershell >>> auto

            python -m venv sklearn-env
            sklearn-env\Scripts\activate  # activate
            pip install -U scikit-plots

          In order to check your installation, you can use:

          .. prompt:: powershell >>> auto

            python -m pip show scikit-plots  # show scikit-plots version and location
            python -m pip freeze             # show all installed packages in the environment
            python -c "import scikitplot; scikitplot.show_versions()"

        .. tab-item:: Linux
          :class-label: tab-4
          :sync: package-manager-pip

          Python 3 is usually installed by default on most Linux distributions. To
          check if you have it installed, try:

          .. prompt:: bash >>> auto

            python3 --version
            pip3 --version

          If you don't have Python 3 installed, please install `python3` and
          `python3-pip` from your distribution's package manager.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install scikit-plots.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: bash >>> auto

            python3 -m venv sklearn-env
            source sklearn-env/bin/activate  # activate
            pip3 install -U scikit-plots

          In order to check your installation, you can use:

          .. prompt:: bash >>> auto

            python3 -m pip show scikit-plots  # show scikit-plots version and location
            python3 -m pip freeze             # show all installed packages in the environment
            python3 -c "import scikitplot; scikitplot.show_versions()"

        .. tab-item:: MacOS
          :class-label: tab-4
          :sync: package-manager-pip

          Install Python 3 using `homebrew <https://brew.sh/>`_ (`brew install python`)
          or by manually installing the package from the `official website
          <https://www.python.org/downloads/macos/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install scikit-plots.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packges.

          .. prompt:: bash >>> auto

            python -m venv sklearn-env
            source sklearn-env/bin/activate  # activate
            pip install -U scikit-plots

          In order to check your installation, you can use:

          .. prompt:: bash >>> auto

            python -m pip show scikit-plots  # show scikit-plots version and location
            python -m pip freeze             # show all installed packages in the environment
            python -c "import scikitplot; scikitplot.show_versions()"

    .. tab-item:: conda
      :class-label: tab-6

      .. include:: installation_instructions_conda.rst


Setting Up Your Environment
---------------------------

To manage dependencies and avoid conflicts, it is recommended to use an
isolated environment, such as `pip venv <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_
or `conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. This approach allows
you to install specific versions of `scikit-learn` and its dependencies
independently from other Python packages.

For Linux users, it is advisable to avoid installing pip packages alongside
those managed by your distribution's package manager (e.g., apt, dnf, pacman).

Activating Your Environment
---------------------------

Remember to activate your chosen environment before running any Python commands,
especially when starting a new terminal session.

Installing Dependencies
-----------------------

If you have not yet installed `NumPy` or `SciPy`, you can do so using `pip`
or `conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.
When using `pip <https://pip.pypa.io/en/stable/>`_, ensure that *binary wheels* are used to avoid
compiling from source. This is particularly important for certain configurations,
such as running Linux on a Raspberry Pi.


Dependencies
============

Mandatory dependencies should be installed automatically if you install Matplotlib using
a package manager such as ``pip`` or ``conda``; therefore this list is primarily for
reference and troubleshooting.

.. grid:: 1 1 2 2
   :class-row: sf-fs-1
   :gutter: 2

   .. grid-item-card::
      :shadow: none

      **Dependencies**
      ^^^
      .. toctree::
         :maxdepth: 2

         dependencies.rst