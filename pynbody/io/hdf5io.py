#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import pickle
import math
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
        self.nz = None
        return self


    def append(self, p):
        self.p.append(p)


    def flush(self):
        with self.fobj as fobj:
            self.snapshot_dumpper(self.p, fobj)
#            self.worldline_dumpper(self.p, fobj)


    def close(self):
        self.fobj.close()


    def store_dset(self, group, name, data):
        olen = len(group[name]) if name in group else 0
        nlen = olen + len(data)
        dset = group.require_dataset(name,
                                     (olen,),
                                     dtype=data.dtype,
                                     maxshape=(None,),
                                     chunks=True,
                                     compression='gzip',
                                     shuffle=True,
                                    )
        dset.resize((nlen,))
        dset[olen:] = data
        return dset


    def snapshot_dumpper(self, p, fobj=None):
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
                name = type(v).__name__.lower()
                data = v.get_state()
                dset = self.store_dset(group, name, data)
                dset.attrs['Class'] = pickle.dumps(type(v))



    def worldline_dumpper(self, p, fobj=None):
        if not p.n: return
        if not isinstance(p, Particles):
            tmp = Particles()
            tmp.append(p)
            p = tmp
        if fobj is None: fobj = self.fobj

        if self.nz is None:
            n = int(p.id.max()+1)
            self.nz = int(math.log10(n))+2

        main_group = fobj.require_group("WorldLines")
        for (key, obj) in p.items:
            if obj.n:
                group = main_group.require_group(key)
                group.attrs['Class'] = pickle.dumps(type(obj))
                for i in np.unique(obj.id):
                    name = "wl"+str(i).zfill(self.nz)
                    data = obj.data[obj.id == i]
                    dset = self.store_dset(group, name, data)



    def dump(self, p, fmode='a'):
        """

        """
        self.setup(fmode)
        with self.fobj as fobj:
            self.dumpper(p, fobj)


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
