# -*- coding: utf-8 -*-
#


"""
This example shows how to setup initial conditions for
a binary star-cluster. Each cluster follows a Plummer
density profile initially in virial equilibrium.
"""

from tupan.units import ureg
from tupan.ics.imf import IMF
from tupan.ics.fewbody import make_binary
from tupan.ics.plummer import make_plummer
from tupan.ics.hierarchy import make_hierarchy
from tupan.io.hdf5io import HDF5IO
from tupan.animation import GLviewer


# Number of particles and softening parameter of each cluster.
#
n1 = 1024
n2 = 4096
eps1 = 4.0 / n1
eps2 = 4.0 / n2


# Initial Mass Function for particles, min/max mass.
#
# imf = IMF.equalmass()
# imf = IMF.salpeter1955(0.1 * ureg.M_sun, 120.0 * ureg.M_sun)
# imf = IMF.padoan2007(0.01 * ureg.M_sun, 120.0 * ureg.M_sun)
# imf = IMF.parravano2011(0.01 * ureg.M_sun, 120.0 * ureg.M_sun)
# imf = IMF.maschberger2013(0.01 * ureg.M_sun, 150.0 * ureg.M_sun)
imf1 = IMF.maschberger2013(0.4 * ureg.M_sun, 4.0 * ureg.M_sun)
imf2 = IMF.maschberger2013(0.1 * ureg.M_sun, 1.0 * ureg.M_sun)

mZAMS1, mtot1 = imf1.sample(n1, seed=1)
mZAMS2, mtot2 = imf2.sample(n2, seed=2)

mtot = mtot1 + mtot2
ureg.define(f'uM = {mtot}')


# Make a Keplerian binary system. This will define
# the center-of-mass coordinates of the clusters.
#
ecc = 0.7
sma = 2.0 * ureg.uL
parent = make_binary(mtot1, mtot2, ecc, sma=sma)


# Now make the hierarchical system, i.e, substitute
# each binary component by a Plummer cluster with
# parameters defined above. Each cluster is created
# with virial radius Rv=1 by default. The final system
# will be rescaled to N-body units.
#
subsys_factory = [
    (make_plummer, n1, eps1, mZAMS1, 1),
    (make_plummer, n2, eps2, mZAMS2, 2),
]
ps = make_hierarchy(parent, subsys_factory)


# Dump the system to disk.
#
with HDF5IO("colliding_clusters", 'w') as fid:
    fid.dump_snap(ps)


# View the final result.
#
with GLviewer() as viewer:
    viewer.show_event(ps)


# -- End of File --
