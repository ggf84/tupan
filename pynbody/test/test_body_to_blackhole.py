#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

if __name__ == "__main__":
    from ggf84decor import selftimer
    from pynbody.models import (IMF, Plummer)
    from pynbody.particles import Particles
    import matplotlib.pyplot as plt
    import numpy as np

    numBodies = 256

    @selftimer
    def main():

#        imf = IMF.equal()
#        imf = IMF.salpeter1955(0.5, 120.0)
#        imf = IMF.parravano2011(0.075, 120.0)
        imf = IMF.padoan2007(0.075, 120.0)

        p = Plummer(numBodies, imf, epsf=4.0, epstype='b', seed=1)
        p.make_plummer()
        p.write_snapshot("plummer"+str(numBodies).zfill(4)+'b')
#        p.show()

        return p.particles.copy()


    p0 = main()

#    p0["body"].vel /= 2

    n_bh = 3

    p = Particles({"body": 1, "blackhole": 1})

    p["body"] = p0["body"][:-n_bh]

    from pynbody.particles.blackhole import dtype

    p["blackhole"].fromlist([tuple(b)+([0.0, 0.0, 0.0],)
                            for b in p0["body"][numBodies-n_bh:]],
                            dtype=dtype)

    p["blackhole"].eps2 *= 0


    from pynbody.io import HDF5IO
    io = HDF5IO("plummer"+str(numBodies-n_bh).zfill(4)+'b'+'-'+str(n_bh)+"bh"+".hdf5")
    io.write_snapshot(p)


########## end of file ##########
