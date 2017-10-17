# -*- coding: utf-8 -*-
#

"""
TODO.
"""


if __name__ == "__main__":
    from tupan.ics.plummer import make_plummer
    from tupan.io.hdf5io import HDF5IO

    n = 256
    eps = 4.0/n

#    imf = ("equalmass",)
#    imf = ("salpeter1955", 0.5, 120.0)
    imf = ("parravano2011", 0.075, 120.0)
#    imf = ("padoan2007", 0.075, 120.0)

    stars = make_plummer(n, eps, imf, seed=1).astype('stars')

    fname = ("plummer" + str(n).zfill(5) + '-' +
             '_'.join(str(i) for i in imf))

    with HDF5IO(fname, 'w') as fid:
        fid.dump_snap(stars)

#    from tupan.ics.fewbody import make_figure83
#    bhs = make_figure83().astype('blackholes')
    from tupan.ics.fewbody import make_pythagorean
    bhs = make_pythagorean().astype('blackholes')

    bhs.dynrescale_total_mass(0.5)
    stars.dynrescale_total_mass(0.5)

    ps = stars + bhs

    ps.reset_pid()
    ps.scale_to_standard()

    fname = ("plummer" + str(stars.n).zfill(5) + '-' +
             '_'.join(str(i) for i in imf) +
             '-' + str(bhs.n) + 'bh')

    with HDF5IO(fname, 'w') as fid:
        fid.dump_snap(ps)

    from tupan.animation import GLviewer
    with GLviewer() as viewer:
        viewer.show_event(ps)


# -- End of File --
