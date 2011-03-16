#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from pynbody.lib.decorators import selftimer
    from pynbody.models import Plummer

    @selftimer
    def main():
        numBodies = 5471
        p = Plummer(numBodies, seed=1)
        p.make_plummer()
        p.write_snapshot()

        bi = p.bodies
        bi.set_acc(bi)

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
