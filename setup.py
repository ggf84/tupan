#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Setup script
"""


import os
from setuptools import setup, find_packages


PATH = os.path.dirname(__file__)


CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Education
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: POSIX :: Linux
Programming Language :: C++
Programming Language :: Python
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Astronomy
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
"""


def version():
    d = {}
    fname = os.path.join(PATH, 'tupan', 'version.py')
    with open(fname) as f:
        exec(f.read(), d)
    return d['VERSION']


def readme():
    fname = os.path.join(PATH, 'README.rst')
    with open(fname) as f:
        return f.read()


setup(
    name='tupan',
    version=version(),
    description="A Python Toolkit for Astrophysical N-Body Simulations.",
    long_description=readme(),
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    url='https://github.com/ggf84/tupan',
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    license='MIT License',
    packages=find_packages(),
    include_package_data=True,
    scripts=['bin/tupan'],
    setup_requires=['cffi>=1.6.0'],
    cffi_modules=[
        'backend_cffi_build.py:ffibuilder32',
        'backend_cffi_build.py:ffibuilder64',
    ],
    install_requires=['cffi>=1.6.0'],
)


# -- End of File --
