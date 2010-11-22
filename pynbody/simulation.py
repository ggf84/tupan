#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Simulation class"""

from __future__ import (print_function, with_statement)
from pprint import pprint
import random
import h5py
import gzip
import time

from .universe import Universe, Body, BlackHole
from vector import Vector


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
                if v['members'] > 0:
                    wl = subgrp[k].create_dataset('wl', (v['members'],3),
                                                  dtype=v['dtype'],
                                                  maxshape=(None,None),
                                                  chunks=True,
                                                  compression='gzip',
                                                  shuffle=True)
#                    tuplefied = []
#                    for kk in sorted(v.keys()):
#                        if isinstance(kk, int):
#                            tuplefied.append(tuple(v.get(kk)))
#                    wl[:,1] = tuplefied


                    tuplefied = [tuple(v.get(kk))
                                 for kk in sorted(v.keys())
                                 if isinstance(kk, int)]
                    wl[:,1] = tuplefied


#                    def myfilter(obj):
#                        return isinstance(obj[0], int)
#                    def func(items):
#                        return tuple(items[1])
#                    tuplefied = map(func, filter(myfilter, sorted(v.items())))
#                    wl[:,1] = tuplefied


#                    pprint(tuplefied)
#                    print(wl[:,1])



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

        for i in xrange(100000):
            data =  (random.randint(16, 64),
                     0.001*random.random(),
                     float(i),
                     random.random(),
                     Vector(random.random(), random.random(), random.random()),
                     Vector(random.random(), random.random(), random.random()))
            self.myuniverse.get_member('body', Body(*data))
        for i in xrange(3):
            data =  (random.randint(16, 64),
                     0.001*random.random(),
                     float(i),
                     random.random(),
                     Vector(random.random(), random.random(), random.random()),
                     Vector(3.0+i, 2.0*i, (i+1)/4.0),
                     Vector(random.random(), random.random(), random.random()))
            self.myuniverse.get_member('bh', BlackHole(*data))

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
#        time.sleep(5)

        for i in range(5):
            t0 = time.time()
            io.dump_snapshot('snap', myuniverse)
            t1 = time.time()
            print(t1-t0)

        f_in = open('snap.hdf5', 'r')
        f_out = gzip.open('snap.hdf5.gz', 'wb')
        f_out.writelines(f_in)
        f_out.close()
        f_in.close()


#        pprint(myuniverse)
        print('running... done')


########## end of file ##########
