#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from pynbody.lib.decorators import selftimer
    from pynbody.models import (IMF, Plummer)
    import matplotlib.pyplot as plt
    import numpy as np

    @selftimer
    def main():
        numBodies = 128

#        imf = IMF.equal()
#        imf = IMF.salpeter1955(0.5, 120.0)
#        imf = IMF.parravano2011(0.075, 120.0)
        imf = IMF.padoan2007(0.075, 120.0)

        p = Plummer(numBodies, imf, epsf=4.0, seed=1)
        p.make_plummer()
        p.write_snapshot()
        p.show()

#        bi = p._body

#        plt.semilogx(np.abs(bi.pos[:,0]),bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.ekin()/bi.mass,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.ekin()/bi.mass+bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),2*bi.ekin()/bi.mass+bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.ekin()/bi.mass-bi.phi,'r,')

#        plt.savefig('evir2.png', bbox_inches="tight")
#        plt.show()



    main()




#    p = Plummer(3, seed=1)
#    p.make_plummer()

#    from pynbody.particles import BlackHole
#    bhdata = BlackHole()
#    bhdata.fromlist([tuple(b)+([0.0, 0.0, 0.0],) for b in p.bodies])

#    from pynbody.io import HDF5IO
#    io = HDF5IO('plummer.hdf5')
#    myuniverse = io.read_snapshot()

#    myuniverse.set_members(bhdata)

#    io = HDF5IO('output.hdf5')
#    io.write_snapshot(myuniverse)










########## end of file ##########
