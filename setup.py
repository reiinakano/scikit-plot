from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys

import scikitplot

here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')
# long_description = ''


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='scikit-plot',
    version=scikitplot.__version__,
    url='https://github.com/reiinakano/scikit-plot',
    license='MIT License',
    author='Reiichiro Nakano',
    tests_require=['pytest'],
    install_requires=[
        'matplotlib>=1.3.1',
        'scikit-learn>=0.18',
        'scipy>=0.9'
    ],
    cmdclass={'test': PyTest},
    author_email='reiichiro.s.nakano@gmail.com',
    description='An intuitive library to add plotting functionality to scikit-learn objects.',
    long_description=long_description,
    packages=['scikitplot'],
    include_package_data=True,
    platforms='any',
    test_suite='scikitplot.tests.test_scikitplot',
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Visualization',
        ],
    extras_require={
        'testing': ['pytest'],
    }
)
