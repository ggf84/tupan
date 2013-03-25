#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function


if __name__ == "__main__":
    from tupan.ics.imf import IMF
    from tupan.ics.plummer import Plummer
    from tupan.io import IO

    def main():
        numBodies = 256

#        imf = IMF.equal()
#        imf = IMF.salpeter1955(0.5, 120.0)
#        imf = IMF.parravano2011(0.075, 120.0)
        imf = IMF.padoan2007(0.075, 120.0)

        p = Plummer(numBodies, imf, eps=4.0/numBodies, eps_parametrization=0, seed=1)
        p.make_plummer()

        fname = "plummer"+str(numBodies).zfill(4)+".hdf5"
        io = IO(fname, 'w')
        io.dump_snapshot(p.particles)

        p.show()

#        import numpy as np
#        import matplotlib.pyplot as plt
#        from tupan.analysis import GLviewer
#        viewer = GLviewer()
#        viewer.initialize()
#        viewer.set_particle(p.particles.copy())
#        viewer.enter_main_loop()


#        bi = p.particles['body']

#        plt.semilogx(np.abs(bi.pos[:,0]),bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.get_ekin()/bi.mass,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.get_ekin()/bi.mass+bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),2*bi.get_ekin()/bi.mass+bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.get_ekin()/bi.mass-bi.phi,'r,')

#        plt.savefig('evir2.png', bbox_inches="tight")
#        plt.show()



    main()


########## end of file ##########
