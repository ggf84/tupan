# -*- coding: utf-8 -*-
#

"""
This example shows how to setup initial conditions for
a star-cluster with blackholes. The cluster follows a
Plummer density profile initially in virial equilibrium.
"""

from tupan.units import ureg
from tupan.ics.imf import IMF
# from tupan.ics.fewbody import make_figure83
from tupan.ics.fewbody import make_pythagorean
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
# imf = IMF.salpeter1955(0.1 * ureg.M_sun, 120.0 * ureg.M_sun)
# imf = IMF.padoan2007(0.01 * ureg.M_sun, 120.0 * ureg.M_sun)
# imf = IMF.parravano2011(0.01 * ureg.M_sun, 120.0 * ureg.M_sun)
imf = IMF.maschberger2013(0.01 * ureg.M_sun, 150.0 * ureg.M_sun)


# Now construct the model in virial equilibrium.
#
mZAMS, mstars = imf.sample(n, seed=1)
mbhs = mstars / 128

mtot = mstars + mbhs
ureg.define(f'uM = {mtot}')

# bhs = make_figure83().astype('blackholes')
bhs = make_pythagorean().astype('blackholes')
bhs.dynrescale_total_mass(mbhs)

ps = make_plummer(n, eps, mZAMS, seed=1)
stars = ps.astype('stars')

ps = stars + bhs

ps.reset_pid()
ps.scale_to_standard()


# Dump the system to disk.
#
fname = ("plummer" + '-n' + str(stars.n) + '-'
         + imf.__name__ + '-' + str(bhs.n) + 'bh')
with HDF5IO(fname, 'w') as fid:
    fid.dump_snap(ps)


# View the final result.
#
with GLviewer() as viewer:
    viewer.show_event(ps)


# -- End of File --
