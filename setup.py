#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas>=0.20.3',
                'bokeh>=0.12.16',
                'scikit-learn>=0.19.1',
                'networkx>=2.1',
                'numpy>=1.13.3']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]


class build_ext(_build_ext):

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


setup(
    author="Feature Labs Team",
    author_email='team@featurelabs.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A collection of utility functions for making demo notebooks.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='henchman',
    name='henchman',
    packages=find_packages(include=['henchman']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/featurelabs/henchman',
    version='0.0.2',
    zip_safe=False,
)
