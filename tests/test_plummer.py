# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from __future__ import print_function


if __name__ == "__main__":
    from tupan.ics.plummer import make_plummer
    from tupan.io import HDF5IO

    n = 256
    eps = 4.0/n

    imf = ("equalmass",)
#    imf = ("salpeter1955", 0.5, 120.0)
#    imf = ("parravano2011", 0.075, 120.0)
#    imf = ("padoan2007", 0.075, 120.0)

    ps = make_plummer(n, eps, imf, seed=1)

    fname = ("plummer" + str(n).zfill(5) + '-' +
             '_'.join(str(i) for i in imf))

    with HDF5IO(fname, 'w') as fid:
        fid.dump_snap(ps)

    from tupan.animation import GLviewer
    viewer = GLviewer()
    viewer.show_event(ps)
    viewer.enter_main_loop()


# -- End of File --
