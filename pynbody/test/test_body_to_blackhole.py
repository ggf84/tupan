#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

if __name__ == "__main__":
    from ggf84decor import selftimer
    from pynbody.models import (IMF, Plummer)
    from pynbody.particles import Particles
    import matplotlib.pyplot as plt
    import numpy as np

    @selftimer
    def main():
        numBodies = 16

#        imf = IMF.equal()
#        imf = IMF.salpeter1955(0.5, 120.0)
#        imf = IMF.parravano2011(0.075, 120.0)
        imf = IMF.padoan2007(0.075, 120.0)

        p = Plummer(numBodies, imf, epsf=4.0, seed=1)
        p.make_plummer()
        p.write_snapshot()
#        p.show()

        return p.particles.copy()


    p0 = main()


    p = Particles({"body": 1, "blackhole": 1})

    p["body"] = p0["body"][:-2]

    from pynbody.particles.blackhole import dtype

    p["blackhole"].fromlist([tuple(b)+([0.0, 0.0, 0.0],) for b in p0["body"][14:]],
                            dtype=dtype)


    from pynbody.io import HDF5IO
    io = HDF5IO('plummer0016bh.hdf5')
    io.write_snapshot(p)


########## end of file ##########
