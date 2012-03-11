#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Script
"""

from distutils.core import setup
import os
import pynbody


package_data = {}
package_data['pynbody.analysis'] = [os.path.join('textures', '*.png')]
package_data['pynbody.lib'] = [os.path.join('ext', '*.c'),
                               os.path.join('ext', '*.h'),
                               os.path.join('ext', '*.cl')]


classifiers = """
Development Status :: 1 - Planning
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python
Topic :: Scientific/Engineering
"""


setup(
    name='PyNbody',
    version=pynbody.version.VERSION,
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    packages=['pynbody',
              'pynbody.analysis',
              'pynbody.integrator',
              'pynbody.io',
              'pynbody.lib',
              'pynbody.lib.utils',
              'pynbody.models',
              'pynbody.particles',
              'pynbody.tests',
             ],
    package_data=package_data,
    scripts=['bin/pynbody'],
    url='http://github.com/GuilhermeFerrari/PyNbody',
    license='MIT License',
    description=pynbody.__doc__.strip(),
    long_description=open('README.txt').read(),
    classifiers=[c for c in classifiers.split('\n') if c],
)


########## end of file ##########
