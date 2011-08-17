#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Script
"""

#from setuptools import setup
from distutils.core import (setup, Extension)
import os
import pynbody


classifiers = """
Development Status :: 1 - Planning
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python
Topic :: Scientific/Engineering
"""

ext_modules = []
path = os.path.join('pynbody', 'lib', 'ext')
ext_modules.append(Extension('pynbody.lib.libgravity',
                             include_dirs = [os.sep+path],
                             libraries = ['m'],
                             sources=[os.path.join(path, 'libgravity.c')]))

package_data = {}
package_data['pynbody.analysis'] = [os.path.join('textures', 'glow.png')]
package_data['pynbody.lib'] = [os.path.join('ext', 'p2p_acc_kernel.cl'),
                               os.path.join('ext', 'p2p_acc_kernel_core.h'),
                               os.path.join('ext', 'p2p_phi_kernel.cl'),
                               os.path.join('ext', 'p2p_phi_kernel_core.h'),
                               os.path.join('ext', 'p2p_acc_kernel_gpugems3.cl')]



setup(
    name='PyNbody',
    version=pynbody.version.VERSION,
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    packages=['pynbody',
              'pynbody.test',
              'pynbody.analysis',
              'pynbody.integrator',
              'pynbody.io',
              'pynbody.lib',
              'pynbody.models',
              'pynbody.particles'],
    ext_modules=ext_modules,
    package_data=package_data,
    scripts=['bin/pynbody'],
    url='http://github.com/GuilhermeFerrari/PyNbody',
    license='MIT License',
    description=pynbody.__doc__.strip(),
    long_description=open('README.txt').read(),
    classifiers=[c for c in classifiers.split('\n') if c],
)


########## end of file ##########
