#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Simulation class"""

from __future__ import (print_function, with_statement)
from pprint import pprint
import random
import h5py

from .universe import Universe, Body, BlackHole


class IO():
    """  """

    def __init__(self):
        self.myuniverse = Universe()

    def dump_snapshot(self, fname='snap', data=None):
        with h5py.File(fname+'.hdf5', 'w') as fobj:
            snap = fobj.create_group('snapshot')
            subgrp = {}
            for (k, v) in data.items():
                subgrp[k] = snap.create_group(k)
                subgrp[k].attrs['dtype'] = str(v['dtype'])
                for (kk, vv) in v.items():
                    if isinstance(kk, int):
                        wl = subgrp[k].create_dataset('worldline'+str(kk), (1,),
                                                      dtype=v['dtype'],
                                                      maxshape=(None,),
                                                      compression='gzip',
                                                      shuffle=True)
                        wl[0] = tuple(vv)



    def load_snapshot(self):
        pass

    def dump_worldlines(self):
        pass

    def read_data(self, fname):
        print('reading data from \'{0}\''.format(fname))
        for i in xrange(10000): a = i*i-2*i+i/5

        for i in xrange(10):
            self.myuniverse.get_member('body', Body(pos=[3.0*i, 2.0+i, (i+1)/2.0]))
        for i in xrange(3):
            self.myuniverse.get_member('bh', BlackHole(vel=[3.0+i, 2.0*i, (i+1)/4.0]))
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

#        print('sleeping...')
#        import time
#        time.sleep(5)

        io.dump_snapshot('snap', myuniverse)

#        pprint(myuniverse)
        print('running... done')


########## end of file ##########
