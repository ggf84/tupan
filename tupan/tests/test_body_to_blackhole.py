# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function

if __name__ == "__main__":
    from tupan.ics.imf import IMF
    from tupan.ics.plummer import Plummer
    from tupan.io import IO

    numBodies = 256

    def main():

#        imf = IMF.equal()
#        imf = IMF.salpeter1955(0.5, 120.0)
#        imf = IMF.parravano2011(0.075, 120.0)
        imf = IMF.padoan2007(0.075, 120.0)

        p = Plummer(numBodies, imf, eps=4.0/numBodies,
                    eps_parametrization=0, seed=1)
        p.make_plummer()

        fname = "plummer"+str(numBodies).zfill(4)+".hdf5"
        io = IO(fname, 'w')
        io.dump_snapshot(p.ps)

#        p.show()

        return p.ps

    p = main()

    from tupan.ics import figure8
    from tupan.particles.blackhole import Blackholes
    b = figure8.make_system()
    p.append(b.astype(Blackholes))

    fname = "plummer"+str(numBodies).zfill(4)+'-'+"3bh"+".hdf5"
    io = IO(fname, 'w')
    io.dump_snapshot(p)


########## end of file ##########
