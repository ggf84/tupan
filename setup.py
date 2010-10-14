from distutils.core import setup
from pynbody import VERSION


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
    version=VERSION,
    author='Guilherme G. Ferrari',
    author_email='gg.ferrari@gmail.com',
    packages=['pynbody', 'pynbody.test'],
    scripts=['scripts/grav.py', 'scripts/sph.py'],
    url='',
    license='MIT License',
    description='A Python Toolkit for Astrophysical N-Body Simulations',
    long_description=open('README.txt').read(),
    classifiers=[c for c in classifiers.split('\n') if c],
)

########## end of file ##########
