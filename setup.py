#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Setup script
"""


import os
from setuptools import setup, find_packages


CWD = os.path.dirname(__file__)


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
    fname = os.path.join(CWD, 'tupan', 'version.py')
    with open(fname) as f:
        exec(f.read(), d)
    return d['VERSION']


def description():
    d = {}
    fname = os.path.join(CWD, 'tupan', '__init__.py')
    with open(fname) as f:
        exec(f.read(), d)
    return d['__doc__']


def readme():
    fname = os.path.join(CWD, 'README.rst')
    with open(fname) as f:
        return f.read()


setup(
    name='tupan',
    version=version(),
    description=description(),
    long_description=readme(),
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    url='https://github.com/ggf84/tupan',
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    license='MIT License',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'tupan = tupan:main',
        ],
    },
    setup_requires=['cffi'],
    cffi_modules=[
        'backend_cffi_build.py:ffibuilder32',
        'backend_cffi_build.py:ffibuilder64',
    ],
    install_requires=['cffi'],
)


# -- End of File --
