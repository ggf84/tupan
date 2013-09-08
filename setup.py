#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Setup Script
"""


import os
from distutils.core import setup

from tupan import version


package_data = {}
package_data['tupan.analysis'] = [os.path.join('textures', '*.png')]
package_data['tupan.lib'] = [os.path.join('src', '*.c'),
                             os.path.join('src', '*.h'),
                             os.path.join('src', '*.cl'),
                             ]


long_description = open(os.path.join(
    os.path.dirname(__file__), 'README.rst')).read()


classifiers = """
Development Status :: 1 - Planning
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python
Topic :: Scientific/Engineering
"""


setup(
    name='Tupan',
    version=version.VERSION,
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    description="A Python Toolkit for Astrophysical N-Body Simulations.",
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
    package_data=package_data,
    scripts=['bin/tupan.py'],
    url='http://github.com/GuilhermeFerrari/Tupan',
    license='MIT License',
    long_description=long_description,
    classifiers=[c for c in classifiers.split('\n') if c],
)


########## end of file ##########
