#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import sys
import pickle
import h5py
from ..lib.utils.timing import decallmethods, timings


__all__ = ["HDF5IO"]

IS_PY3K = True if sys.version_info.major > 2 else False
PICKLE_PROTOCOL = 0 # ensures backward compatibility with Python 2.x

@decallmethods(timings)
class HDF5IO(object):
    """

    """
    def __init__(self, fname, fmode):
        self.fname = fname
        self.fmode = fmode


    def store_dset(self, group, key, obj):
        """

        """
        olen = len(group[key]) if key in group else 0
        dtype = group[key].dtype if key in group else obj.dtype
        nlen = olen + len(obj)
        dset = group.require_dataset(key,
                                     (olen,),
                                     dtype=dtype,
                                     maxshape=(None,),
                                     chunks=True,
                                     compression="gzip",
                                     shuffle=True,
                                    )
        cls = pickle.dumps(type(obj), protocol=PICKLE_PROTOCOL)
        dset.attrs["Class"] = cls.decode('utf-8') if IS_PY3K else cls
        dset.resize((nlen,))
        dset[olen:] = obj.get_state()


    def snapshot_dumpper(self, fobj, p, snap_number):
        """

        """
        base_name = "Snapshot"
        if isinstance(snap_number, int):
            base_name += "_"+str(snap_number).zfill(6)
        base_group = fobj.require_group(base_name)
        group_name = type(p).__name__.lower()
        group = base_group.require_group(group_name)
        cls = pickle.dumps(type(p), protocol=PICKLE_PROTOCOL)
        group.attrs["Class"] = cls.decode('utf-8') if IS_PY3K else cls
        for (key, obj) in p.items():
            if obj.n:
                self.store_dset(group, key, obj)


    def dump_snapshot(self, p, snap_number=None):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            self.snapshot_dumpper(fobj, p, snap_number)


    def snapshot_loader(self, fobj, snap_number=None):
        """

        """
        base_name = "Snapshot"
        if isinstance(snap_number, int):
            base_name += "_"+str(snap_number).zfill(6)
        group = list(fobj[base_name].values())[0]
        p = pickle.loads(group.attrs["Class"])()
        for (k, v) in group.items():
            obj = pickle.loads(v.attrs["Class"])(len(v))
            obj.set_state(v[:])
            p.append(obj)
        return p


    def load_snapshot(self, snap_number=None):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            p = self.snapshot_loader(fobj, snap_number)
        return p


    def worldline_dumpper(self, fobj, wl):
        """

        """
        base_group = fobj.require_group("Worldline")
        group_name = type(wl).__name__.lower()
        group = base_group.require_group(group_name)
        cls = pickle.dumps(type(wl), protocol=PICKLE_PROTOCOL)
        group.attrs["Class"] = cls.decode('utf-8') if IS_PY3K else cls
        for (key, obj) in wl.items():
            if obj.n:
                self.store_dset(group, key, obj)


    def dump_worldline(self, wl):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            self.worldline_dumpper(fobj, wl)


    def worldline_loader(self, fobj, wl_number=None):
        """

        """
        raise NotImplementedError()


    def load_worldline(self, wl_number=None):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            wl = self.worldline_loader(fobj, wl_number)
        return wl


    def to_psdf(self):
        """
        Converts a HDF5 stream into a YAML one.
        """
        from . import psdfio
        fname = self.fname.replace(".hdf5", ".psdf")
        stream = psdfio.PSDFIO(fname)
        p = self.load()
        stream.dump(p, fmode='w')


########## end of file ##########
