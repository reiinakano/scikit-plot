## Makefile
## This Makefile contains various targets for project management tasks such as running the project,
## cleaning up build files, running tests, building Docker images, and more.
## Phony targets are used to avoid conflicts with files of the same name.
## Declare phony targets to indicate these are not files but commands to be executed.
.PHONY: clean examples test publish all
## target: The name of the file or action to be created.
## dependencies: The files that are needed to create the target.
## command: The commands to execute, indented with a tab (not spaces).
# target: dependencies
# 	command

## all target: A convenience target that cleans the build directory and then builds the app.
## Ensures that the project is rebuilt from a clean state.
all: test clean publish
	@echo "all completed."

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean     to remove build artifacts and temporary files"
	@echo "  examples  to execute py scripts under 'examples/' folder"
	@echo "  test      to run unit tests after 'clean'"
	@echo "  publish   to build the project"
	@echo "  all       to run 'test clean publish'"

## get project structure
tree:
	# tree
	find . -type d
	find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"

## Clean up all the generated files
## clean target: Removes build artifacts and cleans up the project directory.
## Useful for ensuring a fresh build environment.
## basic cleaning without 'third_party'
clean_basic:
	@# Command Substitution "$(...)" "(`...`)" (its output in place of the backticks):
	@rm -rf `find -L . -type d -name ".ipynb_checkpoints" -not -path "./third_party/*"`
	@rm -rf "./third_party/.ipynb_checkpoints"
	@echo "Removed all '.ipynb_checkpoints'"
	@rm -rf `find -L . -type d -name "__pycache__" -not -path "./third_party/*"`
	@echo "Removed all '__pycache__'"
	@rm -rf `find -L . -type d -name ".pytest_cache" -not -path "./third_party/*"`
	@echo "Removed all '.pytest_cache'"
	@rm -rf `find -L . -type d -name "__MACOSX" -not -path "./third_party/*"`
	@echo "Removed zip file leftovers '__MACOSX'"
	@echo "basic cleaning completed."

## pypi cleaning in 'build dirs'
clean: clean_basic
	@pip cache purge
	@rm -rf "build" "build_dir" "builddir" "dist" "*.egg-info"
	@echo "Removed folder 'build, egg etc.'"
	@find -L -type f -name "*.so" -path "*/build*"
	@echo "Modules '*.so' files in 'build dirs'"
	@# find -L -type f -name "*.so" | xargs rm -rf
	@find -L -type f -name "*.so" -path "*/build*" -exec rm -rf {} +
	@echo "Removed all '*.so' files in 'build dirs'"
	@echo "pypi cleaning completed."

## test target: Runs pytest on the 'tests/' directory.
## Run this target to execute unit tests.
test: clean_basic
	@cd scikitplot && pytest tests/
	@echo "pytest completed."

## example_script target: Runs py script on the 'examples/' directory.
## Run this target to save generated script plot image.
examples:
	@cd galleries/examples && python calibration/plot_calibration_script.py
	@cd galleries/examples && python classification/plot_classifier_eval_script.py
	@cd galleries/examples && python classification/plot_confusion_matrix_script.py
	@cd galleries/examples && python classification/plot_feature_importances_script.py
	@cd galleries/examples && python classification/plot_learning_curve_script.py
	@cd galleries/examples && python classification/plot_precision_recall_script.py
	@cd galleries/examples && python classification/plot_roc_script.py
	@cd galleries/examples && python clustering/plot_elbow_script.py
	@cd galleries/examples && python clustering/plot_silhouette_script.py
	@cd galleries/examples && python decomposition/plot_pca_2d_projection_script.py
	@cd galleries/examples && python decomposition/plot_pca_component_variance_script.py
	@cd galleries/examples && python kds/plot_cumulative_gain_script.py
	@cd galleries/examples && python kds/plot_ks_statistic_script.py
	@cd galleries/examples && python kds/plot_lift_script.py
	@echo "All py Script executed."

## Builds the pypi Packages
## This target depends on clean and test.
build:
	@## https://mesonbuild.com/meson-python/how-to-guides/editable-installs.html#editable-installs
	@## via 'setup.py' with setuptools
	@## python setup.py build_ext --inplace --verbose
	@# python setup.py sdist
	@# python setup.py bdist_wheel
	@# python setup.py sdist bdist_wheel
	@# python -m pip install --no-build-isolation --no-cache-dir .
	@# python -m pip install --no-build-isolation --no-cache-dir --editable .

	@## Via 'build' or installer need 'pyproject.toml' with setuptools
	@# pip install build
	@# python -m build --sdist
	@# python -m build --wheel
	@# python -m build
	@## python -m pip install --use-pep517 .
	@# python -m pip install --no-build-isolation --no-cache-dir -e .
	@# python -m pip install --no-build-isolation --no-cache-dir -e . -vvv

	@## Via 'build' or installer need 'pyproject.toml' with Meson and Ninja
	@# pip install build meson
	@## Create a build directory
	@# meson setup builddir
	@## Clean previous build artifacts
	@# meson clean -C builddir
	@## Reconfigure the build directory
	@# meson setup --reconfigure builddir
	@# meson setup --wipe builddir
	@## Compile the build directory
	@# meson compile -C builddir
	@# meson compile --clean
	@## Build the project
	@# ninja -C builddir
	@## (Optional) Run tests
	@# ninja -C builddir test
	@python -m pip install --no-build-isolation --no-cache-dir -e .

	@## Create a Tag for the Release
	@git tag -a v0.4.0 -m "Release version 0.4.0"
	@git push origin v1.0.0

	@## Build the PyPI Package
	@git config --global --add safe.directory /home/jovyan/work/contribution/scikit-plots/third_party/NumCpp
	@git submodule sync
	@git submodule update --init --recursive
	@python -m build

	@## *twine* for the upload
	@twine check dist/*
	@twine upload dist/*
	@echo "pypi publish completed."

## Generate a version based on the short commit hash and message
LAST_COMMIT_ID = $(shell git rev-parse --short HEAD)
LAST_COMMIT_MESSAGE = $(shell git log -1 --pretty=%B)

## Tagging the latest commit
tag:
	@echo "Creating tag v$(LAST_COMMIT_ID) with message: $(LAST_COMMIT_MESSAGE)"
	@git tag
	@git tag -a v$(LAST_COMMIT_ID) -m "$(LAST_COMMIT_MESSAGE)"
	@echo "Tag v$(LAST_COMMIT_ID) created with message: $(LAST_COMMIT_MESSAGE)."

## Push the tag to the remote repository
push-tag:
	@echo "Pushing tag v$(LAST_COMMIT_ID) to the remote repository."
	@git push origin v$(LAST_COMMIT_ID)
	@echo "Tag v$(LAST_COMMIT_ID) pushed to the remote repository."

## Release combines tagging and pushing the tag to remote
release: tag push-tag
	@echo "Release v$(LAST_COMMIT_ID) is ready."

# Publish to PyPI (example for Python projects)
publish:
	@echo "Checking the distribution files with twine."
	@twine check dist/*
	@echo "Uploading the distribution files to PyPI."
	@twine upload dist/*
	@echo "PyPI publish completed."

publish_docker:
	# docker login
	## docker tag <existing_image_name>:<existing_tag> <new_image_name>:<new_tag>
	# docker tag jupyter_notebook-base_notebook:latest skplt/scikit-plots-dev:latest
	# docker push skplt/scikit-plots-dev:latest