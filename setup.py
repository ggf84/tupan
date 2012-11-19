#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Script
"""


import os
from distutils.core import setup
from distutils.core import Extension


with open(os.path.join('tupan', 'version.py'), 'r') as fobj:
    exec(fobj.read())


extension_modules = []
path = os.path.join('tupan', 'lib', 'src')
extension_modules.append(Extension('tupan.lib.libc32_gravity',
                             define_macros = [],
                             include_dirs = [os.sep+path],
                             libraries = ['m'],
                             sources=[os.path.join(path, 'libc_gravity.c')]))
extension_modules.append(Extension('tupan.lib.libc64_gravity',
                             define_macros = [("DOUBLE", None)],
                             include_dirs = [os.sep+path],
                             libraries = ['m'],
                             sources=[os.path.join(path, 'libc_gravity.c')]))


package_data = {}
package_data['tupan.analysis'] = [os.path.join('textures', '*.png')]
package_data['tupan.lib'] = [os.path.join('src', '*.c'),
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
    name='Tupan',
    version=VERSION,
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
    ext_modules=extension_modules,
    package_data=package_data,
    scripts=['bin/tupan'],
    url='http://github.com/GuilhermeFerrari/PyNbody',
    license='MIT License',
    long_description=open('README.txt').read(),
    classifiers=[c for c in classifiers.split('\n') if c],
)


########## end of file ##########
