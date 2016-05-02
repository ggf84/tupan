# -*- coding: utf-8 -*-
#


"""
This example shows how to setup initial conditions
for two colliding star-clusters. Each cluster follows
a Plummer density profile initially in virial equilibrium.
"""


import numpy as np
from tupan.io import HDF5IO
from tupan.animation import GLviewer
from tupan.ics.fewbody import make_binary
from tupan.ics.plummer import make_plummer
from tupan.ics.hierarchy import make_hierarchy
np.random.seed(seed=1)


# Mass of each cluster, orbital eccentricity and semi-major-axis.
#
m1 = 0.5
m2 = 0.5
ecc = 0.9
sma = 4.0


# Number of particles per cluster and softening parameter (here N_total=2*n).
#
n = 256
eps = 4.0/(2*n)


# Initial Mass Function for particles, min/max mass.
#
# imf = ("equalmass",)
imf = ("salpeter1955", 0.5, 5.0)
# imf = ("parravano2011", 0.075, 120.0)
# imf = ("padoan2007", 0.075, 120.0)


# Make a Keplerian binary system. This will define
# the center-of-mass coordinates of the clusters.
#
parent = make_binary(m1, m2, ecc, sma=sma)


# Now make the hierarchical system, i.e, substitute
# each binary component by a Plummer cluster with
# parameters defined above. Each cluster is created
# with virial radius Rv=1 by default. The final system
# will be rescaled to N-body units.
#
ps = make_hierarchy(parent, make_plummer, n, eps, imf)


# That's all! Dump the system to disk and view the final result.
#
fname = ("colliding_clusters" + '-n' + str(n) + '-' +
         '_'.join(str(i) for i in imf))

with HDF5IO(fname, 'w') as fid:
    fid.dump_snap(ps)

viewer = GLviewer()
viewer.show_event(ps)
viewer.enter_main_loop()


# -- End of File --
