#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function


if __name__ == "__main__":
    from pynbody.lib.utils.timing import timings
    from pynbody.lib.extensions import kernel_library
    from pynbody.models.imf import IMF
    from pynbody.models.plummer import Plummer
    from pynbody.io import IO
    import matplotlib.pyplot as plt
    import numpy as np

    kernel_library.build_kernels(use_cl=False)

    def main():
        numBodies = 256

#        imf = IMF.equal()
#        imf = IMF.salpeter1955(0.5, 120.0)
#        imf = IMF.parravano2011(0.075, 120.0)
        imf = IMF.padoan2007(0.075, 120.0)

        p = Plummer(numBodies, imf, eps=4.0/numBodies, eps_parametrization=0, seed=1)
        p.make_plummer()

        fname = "plummer"+str(numBodies).zfill(4)
        io = IO(fname, 'hdf5')
        io.dump(p.particles, fmode='w')

        p.show()

#        from pynbody.analysis import GLviewer
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
