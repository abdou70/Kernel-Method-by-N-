#!/usr/bin/env python

from setuptools import setup

setup(
    name='binary',
    version='0.1',
    description='library for working with kernel methods in machine learning',
    packages=['data','helpers'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
    zip_safe=False
    )