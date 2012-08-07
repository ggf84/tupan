#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import pickle
import numpy as np
import h5py
from ..particles import Particles
from ..lib.utils.timing import decallmethods, timings


__all__ = ['HDF5IO']


@decallmethods(timings)
class HDF5IO(object):
    """

    """
    def __init__(self, fname):
        self.fname = fname


    def setup(self, fmode='a'):
        self.fobj = h5py.File(self.fname, fmode)
        self.p = Particles()
        return self


    def append(self, p):
        self.p.append(p)


    def flush(self):
        with self.fobj as fobj:
            self.dumpper(self.p, fobj)


    def close(self):
        self.fobj.close()


    def dumpper(self, p, fobj=None):
        if not p.n: return
        if not isinstance(p, Particles):
            tmp = Particles()
            tmp.append(p)
            p = tmp
        if fobj is None: fobj = self.fobj
        group_name = type(p).__name__.lower()
        group = fobj.require_group(group_name)
        group.attrs['Class'] = pickle.dumps(type(p))
        for (k, v) in p.items:
            if v.n:
                dset_name = type(v).__name__.lower()
                dset_length = 0
                if k in group:
                    dset_length = len(group[k])
                dset = group.require_dataset(dset_name,
                                             (dset_length,),
                                             dtype=v.dtype,
                                             maxshape=(None,),
                                             chunks=True,
                                             compression='gzip',
                                             shuffle=True,
                                            )
                dset.attrs['Class'] = pickle.dumps(type(v))
                olen = len(dset)
                dset.resize((olen+v.n,))
                nlen = len(dset)
                dset[olen:nlen] = v.get_state()


    def dump(self, particles, fmode='a'):
        """

        """
        self.setup(fmode)
        with self.fobj as fobj:
            self.dumpper(particles, fobj)


    def load(self):
        """

        """
        with h5py.File(self.fname, 'r') as fobj:
            group_name = fobj.keys()[0]
            group = fobj.require_group(group_name)
            particles = pickle.loads(group.attrs['Class'])()
            for (k, v) in group.items():
                obj = pickle.loads(v.attrs['Class'])()
                obj.set_state(v[:])
                particles.append(obj)
        return particles


    def to_psdf(self):
        """
        Converts a HDF5 stream into a YAML one.
        """
        from . import psdfio
        fname = self.fname.replace('.hdf5', '.psdf')
        stream = psdfio.PSDFIO(fname)
        p = self.load()
        stream.dump(p, fmode='w')


########## end of file ##########
