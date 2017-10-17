# -*- coding: utf-8 -*-
#


"""
This example shows how to setup initial conditions
for a star-cluster following a Plummer density profile.
The model is constructed in viral equilibrium.
"""

from tupan.units import ureg
from tupan.ics.imf import IMF
from tupan.ics.plummer import make_plummer
from tupan.io.hdf5io import HDF5IO
from tupan.animation import GLviewer


# Number of particles per cluster and softening parameter.
#
n = 256
eps = 4.0 / n


# Initial Mass Function for particles, min/max mass.
#
# imf = IMF.equalmass()
# imf = IMF.salpeter1955(0.4, 120.0)
# imf = IMF.padoan2007(0.075, 120.0)
# imf = IMF.parravano2011(0.075, 120.0)
imf = IMF.maschberger2013(0.01, 150.0)


# Now construct the model in virial equilibrium.
#
mZAMS = imf.sample(n, seed=1) * ureg.M_sun
mtot = mZAMS.sum()
ureg.define(f'uM = {mtot}')
ps = make_plummer(n, eps, mZAMS, seed=1)


# Dump the system to disk and view the final result.
#

fname = "plummer" + '-n' + str(n) + '-' + imf.__name__
with HDF5IO(fname, 'w') as fid:
    fid.dump_snap(ps)

with GLviewer() as viewer:
    viewer.show_event(ps)


# -- End of File --
