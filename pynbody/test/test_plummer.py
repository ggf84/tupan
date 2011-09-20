#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from pynbody.models import (IMF, Plummer)
    import matplotlib.pyplot as plt
    import numpy as np

    def main():
        numBodies = 256

#        imf = IMF.equal()
#        imf = IMF.salpeter1955(0.5, 120.0)
#        imf = IMF.parravano2011(0.075, 120.0)
        imf = IMF.padoan2007(0.075, 120.0)

        p = Plummer(numBodies, imf, epsf=4.0, epstype='b', seed=1)
        p.make_plummer()
        p.write_snapshot("plummer"+str(numBodies).zfill(4)+'b')
        p.show()

#        bi = p._body

#        plt.semilogx(np.abs(bi.pos[:,0]),bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.get_ekin()/bi.mass,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.get_ekin()/bi.mass+bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),2*bi.get_ekin()/bi.mass+bi.phi,'r,')
#        plt.semilogx(np.abs(bi.pos[:,0]),bi.get_ekin()/bi.mass-bi.phi,'r,')

#        plt.savefig('evir2.png', bbox_inches="tight")
#        plt.show()



    main()


########## end of file ##########
