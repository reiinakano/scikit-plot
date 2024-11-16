# setup.py
"""
# conf.py

# Add the extension to the Sphinx extensions list
extensions = ["sphinx_extensions.infer_next_release_versions"]

# Configuration for the version switcher JSON URL
html_theme_options = {
    "switcher": {
        "json_url": "https://your-url-to-versions-json",
    }
}
"""

from setuptools import setup, find_packages

setup(
    name="version_info_extension",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "sphinx",
        "scikit-plots",
        "packaging"
    ],
    entry_points={
        "sphinx.extension": [
            "version_info_extension = sphinx_extensions.version_info_extension",
        ],
    },
    author="Your Name",
    description="A Sphinx extension to infer the next release versions",
    classifiers=[
        "Framework :: Sphinx",
        "Programming Language :: Python :: 3",
    ],
)
