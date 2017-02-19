## Steps for contributing

Fixing a bug you found in Scikit-plot? Suggesting a feature? Adding your own plotting function? Listed here are some guidelines to keep in mind when contributing.

1. **Open an issue** along with detailed explanation. For bug reports, include the code to reproduce the bug. For feature requests, explain why you think the feature could be useful.

2. **Fork the repository**. If you're contributing code, clone the forked repository into your local machine.

3. **Run the tests** to make sure they pass on your machine. Simply run `pytest` at the root folder and make sure all tests pass.

4. **Create a new branch**. Please do not commit directly to the master branch. Create your own branch and place your additions there.

5. **Write your code**. Please follow PEP8 coding standards. Also, if you're adding a function, you must [write a docstring using the Google format](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) detailing the API of your function. Take a look at the docstrings of the other Scikit-plot functions to get an idea of what the docstring of yours should look like.

6. **Write/modify the corresponding unit tests**. After adding in your code and the corresponding unit tests, run `pytest` again to make sure they pass.

7. **Submit a pull request**. After submitting a PR, if all tests pass, your code will be reviewed and merged promptly.

Thank you for taking the time to make Scikit-plot better!