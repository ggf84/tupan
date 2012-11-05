#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import pickle
import h5py
import numpy as np
from ..lib.utils.timing import decallmethods, timings


__all__ = ["HDF5IO"]


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
        nlen = olen + len(obj)
        dset = group.require_dataset(key,
                                     (olen,),
                                     dtype=obj.dtype,
                                     maxshape=(None,),
                                     chunks=True,
                                     compression="gzip",
                                     shuffle=True,
                                    )
        dset.attrs["Class"] = pickle.dumps(type(obj))
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
        group.attrs["Class"] = pickle.dumps(type(p))
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
        group = fobj[base_name].values()[0]
        p = pickle.loads(group.attrs["Class"])()
        for (k, v) in group.items():
            obj = pickle.loads(v.attrs["Class"])()
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
        group.attrs["Class"] = pickle.dumps(type(wl))
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
