#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['pandas>=0.20.3',
                'bokeh>=0.12.16',
                'scikit-learn>=0.19.1',
                'networkx>=2.1',
                'scipy>=1.0.0',
                'numpy>=1.13.3',
                'ipython>=5.4']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Feature Labs Team",
    author_email='team@featurelabs.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    description="A collection of utility functions for making demo notebooks.",
    install_requires=requirements,
    license="BSD 3-clause",
    long_description=readme,
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
