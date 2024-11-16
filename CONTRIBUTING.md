## Steps for contributing

Fixing a bug you found in Scikit-plot? Suggesting a feature? Adding your own plotting function? Listed here are some guidelines to keep in mind when contributing.

1. **Open an issue** along with detailed explanation. For bug reports, include the code to reproduce the bug. For feature requests, explain why you think the feature could be useful.

2. **Fork the repository**. If you're contributing code, clone the forked repository into your local machine.

- If you are a first-time contributor:
    - Go to github and click the “fork” button to create your own copy of the project.
    - Clone the project to your local computer:
        ```bash
        git clone --recurse-submodules https://github.com/your-username/scikit-plot.git
        ```
    - Now, `git remote -v` will show two remote repositories named:
        - upstream, which refers to the scikit-plot repository
        - origin, which refers to your personal fork
    - Pull the latest changes from upstream, including tags:
        ```bash
        git checkout main
        git pull upstream main --tags
        ```
    - Initialize submodules:
        ```bash
        git submodule update --init
        ```
3. **Run the tests** to make sure they pass on your machine. Simply run `pytest` at the root folder and make sure all tests pass.

4. **Create a new branch**. Please do not commit directly to the master branch. Create your own branch and place your additions there.

- Develop your contribution:
    - Create a branch for the feature you want to work on. Since the branch name will appear in the merge message, use a sensible name such as 'linspace-speedups':
        ```bash
        git checkout -b linspace-speedups
        ```
5. **Write your code**. Please follow PEP8 coding standards. Also, if you're adding a function, you must currently [write a docstring using the Google format](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) detailing the API of your function or In Feature [NumPy docstring standard](https://numpy.org/devdocs/dev/howto-docs.html#howto-document). Take a look at the docstrings of the other Scikit-plot functions to get an idea of what the docstring of yours should look like.

- Commit locally as you progress (`git add` and `git commit`) Use a properly formatted commit message, write tests that fail before your change and pass afterward, run all the tests locally. Be sure to document any changed behavior in docstrings, keeping to the NumPy docstring [standard](https://numpy.org/devdocs/dev/howto-docs.html#howto-document).

6. **Write/modify the corresponding unit tests**. After adding in your code and the corresponding unit tests, run `pytest` again to make sure they pass.

7. **Submit a pull request**. After submitting a PR (pull requests), if all tests pass, your code will be reviewed and merged promptly.

- To submit your contribution:
    - Push your changes back to your fork on GitHub:
        ```bash
        git push origin linspace-speedups
        ```
    - Go to GitHub. The new branch will show up with a green Pull Request button. Make sure the title and message are clear, concise, and self- explanatory. Then click the button to submit it.

    - If your commit introduces a new feature or changes functionality, post on the mailing list to explain your changes. For bug fixes, documentation updates, etc., this is generally not necessary, though if you do not get any reaction, do feel free to ask for review.
- Review process:

Thank you for taking the time to make Scikit-plot better!