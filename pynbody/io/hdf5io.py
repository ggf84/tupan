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
    def __init__(self, fname, fmode='a'):
        self.fname = fname
        self.fmode = fmode


    def setup(self):
        self.wl = WorldLine()
        return self


    def append(self, p):
        self.wl.append(p)
        self.flush(128)


    def flush(self, size=1):
        if self.wl.n >= size:
#            print('flush', self.wl.n)
            self.dump_worldline(self.wl, size)


    def close(self):
        self.fobj.close()


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


    def dump_snapshot(self, p, snap_number=None):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            self.snapshot_dumpper(fobj, p, snap_number)


    def snapshot_dumpper(self, fobj, p, snap_number):
        base_name = "Snapshot"
        if isinstance(snap_number, int):
            base_name += "_"+str(snap_number).zfill(6)
        base_group = fobj.require_group(base_name)
        group_name = type(p).__name__.lower()
        group = base_group.require_group(group_name)
        group.attrs['Class'] = pickle.dumps(type(p))
        for (key, obj) in p.items():
            if obj.n:
                name = key
                data = obj.data
                self.store_dset(group, name, data, data.dtype)


    def dump_worldline(self, wl, size=1):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            self.worldline_dumpper(fobj, wl, size)


    def worldline_dumpper(self, fobj, wl, size=1):
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


#    def load(self):
#        """
#
#        """
#        with h5py.File(self.fname, 'r') as fobj:
#            group_name = fobj.keys()[0]
#            group = fobj.require_group(group_name)
#            particles = pickle.loads(group.attrs['Class'])()
#            for (k, v) in group.items():
#                obj = type(particles.kind[k])()
#                obj.set_state(v[:])
#                particles.append(obj)
#        return particles


    def load_worldline(self, wl_number):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            pass
        raise NotImplementedError()


    def load_snapshot(self, snap_number=None):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            base_name = "Snapshot"
            if isinstance(snap_number, int):
                base_name += "_"+str(snap_number).zfill(6)
            group = fobj[base_name]["particles"]
            p = pickle.loads(group.attrs['Class'])()
            for (k, v) in group.items():
                obj = type(p.kind[k])()
                obj.set_state(v[:])
                p.append(obj)
        return p


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
