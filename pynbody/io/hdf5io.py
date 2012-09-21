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
from ..particles.allparticles import Body, BlackHole, Sph
from ..lib.utils.timing import decallmethods, timings


__all__ = ['HDF5IO']


#@decallmethods(timings)
#class WorldLine(object):
#    """
#
#    """
#    def __init__(self):
#        self.wl = {'body': {'name': [], 'data': [], 'dtype': None},
#                   'blackhole': {'name': [], 'data': [], 'dtype': None},
#                   'sph': {'name': [], 'data': [], 'dtype': None}
#                  }
#
#    def append(self, p):
#        if p.n:
#            for (key, obj) in p.items:
#                if obj.n:
#                    self.wl[key]['dtype'] = obj.data.dtype
#                    for d in obj.data:
#                        name = "wl"+str(int(d['id'])).zfill(4)
#                        data = d.item()
#
#                        if name in self.wl[key]['name']:
#                            index = self.wl[key]['name'].index(name)
#                            self.wl[key]['data'][index].append(data)
#                        else:
#                            self.wl[key]['name'].append(name)
#                            self.wl[key]['data'].append([data])
#
#
#    def worldline_dumpper(self, wl, fobj=None):
#        if fobj is None: fobj = self.fobj
#
#        main_group = fobj.require_group("WorldLines")
#
#        for (key, d) in wl.wl.items():
#            group = main_group.require_group(key)
#            for (name, data) in zip(d['name'], d['data']):
#                dset = self.store_dset(group, name, data, d['dtype'])
#
#
#
#
#    @property
#    def min_size(self):
#        size = 64
#        for (key, d) in self.wl.items():
#            for (name, data) in zip(d['name'], d['data']):
#                if len(data) < size:
#                    size = len(data)
#        return size


@decallmethods(timings)
class HDF5IO(object):
    """

    """
    def __init__(self, fname):
        self.fname = fname


    def setup(self):
        self.p = Particles()
        self.nz = None
        return self


    def append(self, p, buffer_length=32768):
        if self.nz is None:
            n = int(p.id.max()+1)
            self.nz = int(math.log10(n))+2

        self.p.append(p)
        if self.p.n >= buffer_length:
            print('append', self.p.n)
            self.dump(self.p)
            self.p = Particles()


    def flush(self):
        if self.p.n:
            print('flush', self.p.n)
            self.dump(self.p)
            self.p = Particles()



    def dump(self, p, fmode='a'):
        """

        """
        with h5py.File(self.fname, fmode) as fobj:
            self.snapshot_dumpper(p, fobj)
#            self.worldline_dumpper(p, fobj)



    def store_dset(self, group, name, data, dtype):
        olen = len(group[name]) if name in group else 0
        nlen = olen + len(data)
        dset = group.require_dataset(name,
                                     (olen,),
                                     dtype=dtype,
                                     maxshape=(None,),
                                     chunks=True,
                                     compression='gzip',
                                     shuffle=True,
                                    )
        dset.resize((nlen,))
        dset[olen:] = data
        return dset


    def snapshot_dumpper(self, p, fobj):
        group_name = type(p).__name__.lower()
        group = fobj.require_group(group_name)
        group.attrs['Class'] = pickle.dumps(type(p))
        for (key, obj) in p.items:
            if obj.n:
                name = type(obj).__name__.lower()
                data = obj.get_state()
                dset = self.store_dset(group, name, data, data.dtype)
                dset.attrs['Class'] = pickle.dumps(type(obj))


    def worldline_dumpper(self, p, fobj):
        main_group = fobj.require_group("WorldLines")
        for (key, obj) in p.items:
            if obj.n:
                group = main_group.require_group(key)
                group.attrs['Class'] = pickle.dumps(type(obj))
                for i in np.unique(obj.id):
                    name = "wl"+str(i).zfill(self.nz)
                    data = obj.data[obj.id == i]
                    dset = self.store_dset(group, name, data, data.dtype)




    def close(self):
        self.fobj.close()


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
