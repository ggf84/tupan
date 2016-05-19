# -*- coding: utf-8 -*-
#


"""
This example shows how to setup initial conditions
for a star-cluster following a Plummer density profile.
The model is constructed in viral equilibrium.
"""


from tupan.io import HDF5IO
from tupan.animation import GLviewer
from tupan.ics.plummer import make_plummer


# Number of particles per cluster and softening parameter.
#
n = 256
eps = 4.0/n


# Initial Mass Function for particles, min/max mass.
#
# imf = ("equalmass",)
imf = ("salpeter1955", 0.5, 5.0)
# imf = ("parravano2011", 0.075, 120.0)
# imf = ("padoan2007", 0.075, 120.0)


# Now construct the model in virial equilibrium.
#
ps = make_plummer(n, eps, imf, seed=1)


# That's all! Dump the system to disk and view the final result.
#
fname = ("plummer" + '-n' + str(n) + '-' +
         '_'.join(str(i) for i in imf))

with HDF5IO(fname, 'w') as fid:
    fid.dump_snap(ps)

viewer = GLviewer()
viewer.show_event(ps)
viewer.enter_main_loop()


# -- End of File --
