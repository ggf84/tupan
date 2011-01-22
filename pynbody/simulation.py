#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Simulation class"""

from __future__ import print_function
from pprint import pprint
import h5py
import gzip
import time

from .universe import (Universe, Body, BlackHole)
from .vector import Vector

from .models import Plummer


class IO():
    """  """

    def __init__(self):
        self.myuniverse = Universe()

    def dump_snapshot(self, fname='snap', data=None):
        with h5py.File(fname+'.hdf5', 'w') as fobj:
            snap = fobj.create_group('snapshot')
            subgrp = {}
            for (k, v) in data.items():
                subgrp = snap.create_group(k)
                if len(v['array']) > 0:
                    subgrp.attrs['format'] = str(v['format'])
                    wl = subgrp.create_dataset('wl', (len(v['array']),3),
                                               dtype=v['format'],
                                               maxshape=(None,None),
                                               chunks=True,
                                               compression='gzip',
                                               shuffle=True)
                    if isinstance(v['array'], list):
                        wl[:,1] = [tuple(b) for b in v['array']]
                    else:
                        wl[:,1] = v['array'].get_data()


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

        p = Plummer(16, seed=1)
        p.make_plummer()
        self.myuniverse.set_members('body', p.bodies)

        p = Plummer(4, seed=1)
        p.make_plummer()
        bhdata = [BlackHole(tuple(b)+(Vector(0.0, 0.0, 0.0),)) for b in p.bodies._array.tolist()]
        self.myuniverse.set_members('bh', bhdata)

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


        for i in range(5):
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
