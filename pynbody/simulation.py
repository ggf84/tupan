#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Simulation class"""

from __future__ import print_function
from pprint import pprint
import h5py
import gzip
import time

from .models import Plummer
from .particles import (Particles, BlackHoles)


class IO():
    """  """

    def __init__(self):
        self.myuniverse = Particles()

    def dump_snapshot(self, fname='snap', data=None):
        with h5py.File(fname+'.hdf5', 'w') as fobj:
            snap = fobj.create_group('snapshot')
            for (k, v) in data.items():
                subgrp = snap.create_group(k)
                if len(v) > 0:
                    subgrp.attrs['dtype'] = str(v.array.dtype)
                    wl = subgrp.create_dataset('wl', (len(v),3),
                                               dtype=v.array.dtype,
                                               maxshape=(None,None),
                                               chunks=True,
                                               compression='gzip',
                                               shuffle=True)
                    wl[:,1] = v.get_data()


#        with h5py.File(fname+'.hdf5', 'w') as fobj:
#            snap = fobj.create_group('snapshot')
#            subgrp = {}
#            for (k, v) in data.items():
#                subgrp[k] = snap.create_group(k)
#                subgrp[k].attrs['dtype'] = str(v['dtype'])
#                for (kk, vv) in v.items():
#                    if isinstance(kk, int):
#                        wl = subgrp[k].create_dataset('wl'+str(kk), (1,),
#                                                      dtype=v['dtype'],
#                                                      maxshape=(None,),
#                                                      chunks=True,
#                                                      compression='gzip',
#                                                      shuffle=True)
#                        wl[0] = tuple(vv)


    def load_snapshot(self):
        pass

    def dump_worldlines(self):
        pass

    def read_data(self, fname):
        print('reading data from \'{0}\''.format(fname))

        p = Plummer(5471, seed=1)
        p.make_plummer()
        self.myuniverse.set_member(p.bodies)

        p = Plummer(3, seed=1)
        p.make_plummer()
        bhdata = [tuple(b)+([0.0, 0.0, 0.0],) for b in p.bodies.array.tolist()]
        bh = BlackHoles()
        bh.fromlist(bhdata)
        self.myuniverse.set_member(bh)

        return self.myuniverse

    def take_a_particle_based_sampling(self):
        pass


class Diagnostic():
    """  """

    def write_diagnostic(self):
        pass


class Simulation(IO):
    """The Simulation class is the top level class for N-body simulations"""

    def __init__(self, args):
        self.time = 0.0           # global simulation time
        self.eta = args.tau
        self.fdia = args.dia      # diagnostic frequency per time unit
        self.fdump = args.out     # indicates how often each particle is dumped
        self.tend = args.end
        self.icfname = args.ic
        self.sim_mode = args.mod


    def prepare_for_run(self):
        pass


    def run(self):
        """Initialize a N-body simulation"""
        print('running...')
        io = IO()
        myuniverse = io.read_data(self.icfname)


        for i in range(1):
            t0 = time.time()
            io.dump_snapshot('snap', myuniverse)
            t1 = time.time()
            print(t1-t0)

#        f_in = open('snap.hdf5', 'r')
#        f_out = gzip.open('snap.hdf5.gz', 'wb')
#        f_out.writelines(f_in)
#        f_out.close()
#        f_in.close()


        pprint(myuniverse)
        print('running... done')


########## end of file ##########
