#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Script
"""

#from setuptools import setup
from distutils.core import setup
from distutils.command.install import USER_SITE
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


data_files = {}

path = os.path.join('pynbody', 'lib', 'kernels') + os.sep
installpath = USER_SITE + os.sep + path
data_files[installpath] = [path+fname for fname in ['p2p_pot_kernel.cl',
                                                    'p2p_acc_kernel.cl',
                                                    'p2p_acc_kernel_gpugems3.cl']]
path = os.path.join('pynbody', 'analysis', 'textures') + os.sep
installpath = USER_SITE + os.sep + path
data_files[installpath] = [path+fname for fname in ['bh.png',
                                                    'star.png']]



setup(
    name='PyNbody',
    version=pynbody.version,
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    packages=['pynbody',
              'pynbody.analysis',
              'pynbody.integrator',
              'pynbody.io',
              'pynbody.lib',
              'pynbody.lib.kernels',
              'pynbody.models',
              'pynbody.particles',
              'pynbody.test'],
#    include_package_data=True,
    data_files=data_files.items(),
    scripts=['bin/main.py'],
    url='http://github.com/GuilhermeFerrari/PyNbody',
    license='MIT License',
    description=pynbody.__doc__.strip(),
    long_description=open('README.txt').read(),
    classifiers=[c for c in classifiers.split('\n') if c],
)

########## end of file ##########
