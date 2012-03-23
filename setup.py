#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Script
"""

from distutils.core import setup
from distutils.core import Extension
import os
import pynbody


extension_modules = []
path = os.path.join('pynbody', 'lib', 'src')
extension_modules.append(Extension('pynbody.lib.libc32_gravity',
                             define_macros = [],
                             include_dirs = [os.sep+path],
                             libraries = ['m'],
                             sources=[os.path.join(path, 'libc_gravity.c')]))
extension_modules.append(Extension('pynbody.lib.libc64_gravity',
                             define_macros = [("DOUBLE", None)],
                             include_dirs = [os.sep+path],
                             libraries = ['m'],
                             sources=[os.path.join(path, 'libc_gravity.c')]))

package_data = {}
package_data['pynbody.analysis'] = [os.path.join('textures', '*.png')]
package_data['pynbody.lib'] = [os.path.join('src', '*.c'),
                               os.path.join('src', '*.h'),
                               os.path.join('src', '*.cl')]

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
    ext_modules=extension_modules,
    package_data=package_data,
    scripts=['bin/pynbody'],
    url='http://github.com/GuilhermeFerrari/PyNbody',
    license='MIT License',
    description=pynbody.__doc__.strip(),
    long_description=open('README.txt').read(),
    classifiers=[c for c in classifiers.split('\n') if c],
)


########## end of file ##########
