#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

if __name__ == "__main__":
    from pynbody.models.imf import IMF
    from pynbody.models.plummer import Plummer
    from pynbody.particles import Particles
    from pynbody.io import IO
    import matplotlib.pyplot as plt
    import numpy as np

    numBodies = 256

    def main():

#        imf = IMF.equal()
#        imf = IMF.salpeter1955(0.5, 120.0)
#        imf = IMF.parravano2011(0.075, 120.0)
        imf = IMF.padoan2007(0.075, 120.0)

        p = Plummer(numBodies, imf, eps=4.0/numBodies, eps_parametrization=0, seed=1)
        p.make_plummer()

        fname = "plummer"+str(numBodies).zfill(4)
        io = IO(fname, 'hdf5')
        io.dump(p.particles, fmode='w')

#        p.show()

        return p.particles.copy()


    p0 = main()

#    p0["body"].vel /= 2

    n_bh = 3

    p = Particles({"body": 1, "blackhole": 1})

    p["body"] = p0["body"][:-n_bh]

    from pynbody.particles.blackhole import dtype

    p["blackhole"].fromlist([tuple(b)+([0.0, 0.0, 0.0],[0.0, 0.0, 0.0],)
                            for b in p0["body"][numBodies-n_bh:]],
                            dtype=dtype)

    p["blackhole"].eps2 *= 0


    fname = "plummer"+str(numBodies-n_bh).zfill(4)+'-'+str(n_bh)+"bh"
    io = IO(fname, 'hdf5')
    io.dump(p, fmode='w')


########## end of file ##########
