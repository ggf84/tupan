# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function


if __name__ == "__main__":
    from tupan.ics.plummer import make_plummer
    from tupan.io import IO

    n = 256
    eps = 4.0/n

    imf = ("equalmass",)
#    imf = ("salpeter1955", 0.5, 120.0)
#    imf = ("parravano2011", 0.075, 120.0)
#    imf = ("padoan2007", 0.075, 120.0)

    ps = make_plummer(n, eps, imf, seed=1)

    fname = ("plummer" + str(n).zfill(5) + '-'
             + '_'.join(str(i) for i in imf) + ".hdf5")
    io = IO(fname, 'w')
    io.dump_snapshot(ps)

#     p.show()

    from tupan.analysis.glviewer import GLviewer
    viewer = GLviewer()
    viewer.initialize()
    viewer.set_particle_system(ps)
    viewer.enter_main_loop()


# -- End of File --
