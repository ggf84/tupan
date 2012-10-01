#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import pickle
import math
from collections import defaultdict
import numpy as np
import h5py
from ..lib.utils.timing import decallmethods, timings


__all__ = ['HDF5IO']


@decallmethods(timings)
class WorldLine(object):
    """

    """
    def __init__(self):
#        self.kind = defaultdict(list)
        self.kind = defaultdict(dict)
        self.dtype = {}
        self.nz = None
        self.cls = None
        self.clear()


    def clear(self):
        self.n = 0
        self.kind.clear()


    def append(self, p):
        if self.cls is None: self.cls = type(p)
        if self.nz is None: self.nz = int(math.log10(p.n+1))+2

#        for (key, obj) in p.items():
#            if obj.n:
#                data = obj.data.copy()
#                self.kind[key].append(data)
#                self.n += len(data)


        self.n = 0
        for (key, obj) in p.items():
            if obj.n:
                self.dtype[key] = obj.data.dtype
                d = self.kind[key]
                for data in obj.data.copy():
                    i = data[0]
                    d.setdefault(i, []).append(data)
                    self.n = max(self.n, len(d[i]))


    def keys(self):
        return self.kind.keys()


    def objs(self):
        return self.kind.values()


    def items(self):
        return self.kind.items()




@decallmethods(timings)
class HDF5IO(object):
    """

    """
    def __init__(self, fname):
        self.fname = fname


    def setup(self):
        self.wl = WorldLine()
        return self


    def append(self, p):
        self.wl.append(p)
        self.flush(128)


    def flush(self, size=1):
        if self.wl.n >= size:
#            print('flush', self.wl.n)
            self.dump(self.wl, size)


    def dump(self, wl, size=1, fmode='a'):
        """

        """
        with h5py.File(self.fname, fmode) as fobj:
#            self.snapshot_dumpper(wl, fobj)
#            self.worldline_dumpper(wl, fobj)
            self.worldline_dumpper(wl, fobj, size)
#        wl.clear()


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


    def snapshot_dumpper(self, wl, fobj):
        base_group = fobj.require_group("Snapshots")
        group_name = wl.cls.__name__.lower()
        group = base_group.require_group(group_name)
        group.attrs['Class'] = pickle.dumps(wl.cls)
        for (k, v) in wl.items():
            if v:
                name = k
                data = np.concatenate(v)
                self.store_dset(group, name, data, data.dtype)


#    def worldline_dumpper(self, wl, fobj):
#        base_group = fobj.require_group("Worldlines")
#        main_group_name = wl.cls.__name__.lower()
#        main_group = base_group.require_group(main_group_name)
#        main_group.attrs['Class'] = pickle.dumps(wl.cls)
#        for (k, v) in wl.items():
#            if v:
#                group = main_group.require_group(k)
#                d = {}
#                for obj in v:
#                    for o in obj:
#                        name = "wl"+str(int(o['id'])).zfill(wl.nz)
#                        data = [o]
#                        d.setdefault(name, type(data)()).extend(data)
#                for (name, data) in d.items():
#                    dtype = data[0].dtype
#                    self.store_dset(group, name, data, dtype)


    def worldline_dumpper(self, wl, fobj, size=1):
        base_group = fobj.require_group("Worldlines")
        main_group_name = wl.cls.__name__.lower()
        main_group = base_group.require_group(main_group_name)
        main_group.attrs['Class'] = pickle.dumps(wl.cls)
        for (k, d) in wl.items():
            if d:
                dtype = wl.dtype[k]
                group = main_group.require_group(k)
                for (i, data) in d.items():
                    if len(data) >= size:
                        name = "wl"+str(i).zfill(wl.nz)
                        self.store_dset(group, name, data, dtype)
                        d[i] = []


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
                obj = type(particles.kind[k])()
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
