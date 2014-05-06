#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Setup Script
"""


import os
from distutils.core import setup

from tupan import version


PACKAGE_DATA = {}
PACKAGE_DATA['tupan'] = ['tupan.cfg']
PACKAGE_DATA['tupan.analysis'] = [os.path.join('textures', '*.png')]
PACKAGE_DATA['tupan.lib'] = [os.path.join('src', '*.c'),
                             os.path.join('src', '*.h'),
                             os.path.join('src', '*.cl'),
                             ]


LONG_DESCRIPTION = open(os.path.join(
    os.path.dirname(__file__), 'README.rst')).read()


CLASSIFIERS = """
Development Status :: 1 - Planning
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python
Topic :: Scientific/Engineering
"""


setup(
    name='tupan',
    version=version.VERSION,
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    description="A Python Toolkit for Astrophysical N-Body Simulations.",
    long_description=LONG_DESCRIPTION,
    packages=['tupan',
              'tupan.analysis',
              'tupan.ics',
              'tupan.integrator',
              'tupan.io',
              'tupan.lib',
              'tupan.lib.utils',
              'tupan.particles',
              'tupan.tests',
              ],
    package_data=PACKAGE_DATA,
    scripts=['bin/tupan-simulation.py'],
    url='https://github.com/ggf84/tupan',
    license='MIT License',
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
)


# -- End of File --
