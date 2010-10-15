#!/usr/bin/env python

"""
Setup Script
"""

from distutils.core import setup
import pynbody


CLASSIFIERS = """
Development Status :: 1 - Planning
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python
Topic :: Scientific/Engineering
"""


setup(
    name='PyNbody',
    version=pynbody.VERSION,
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    packages=['pynbody', 'pynbody.test'],
    scripts=['bin/main.py'],
    url='http://github.com/GuilhermeFerrari/PyNbody',
    license='MIT License',
    description=pynbody.__doc__.strip(),
    long_description=open('README.txt').read(),
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
)

########## end of file ##########
