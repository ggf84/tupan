#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import pickle
import h5py
from pynbody.lib.utils.timing import timings


__all__ = ['HDF5IO']


class HDF5IO(object):
    """

    """
    def __init__(self, fname, fmode='a'):
        if not fname.endswith(".hdf5"):
            fname += ".hdf5"
        self.fname = fname
        self.fmode = fmode


    @timings
    def write_snapshot(self, data, snap_name=None, snap_time=None):
        """

        """
        with h5py.File(self.fname, self.fmode) as fobj:
            data_name = data.__class__.__name__
            if isinstance(snap_name, str):
                snap_grp = fobj.require_group(snap_name)
                snap_grp.attrs['Time'] = pickle.dumps(snap_time)
                data_grp = snap_grp.require_group(data_name)
            else:
                data_grp = fobj.require_group(data_name)
            data_grp.attrs['Class'] = pickle.dumps(data.__class__)
            for (k, v) in data.items():
                if v:
                    dset_name = v.__class__.__name__
                    dset = data_grp.require_dataset(dset_name,
                                                    (len(v),),
                                                    dtype=v._dtype,
                                                    maxshape=(None,),
                                                    chunks=True,
                                                    compression='gzip',
                                                    shuffle=True)
                    dset.attrs['Class'] = pickle.dumps(v.__class__)
                    dset[:] = v.get_data()


    @timings
    def read_snapshot(self, snap_name=None):
        """

        """
        with h5py.File(self.fname, 'r') as fobj:
            if isinstance(snap_name, str):
                snap_grp = fobj.require_group(snap_name)
                snap_time = pickle.loads(snap_grp.attrs['Time'])
                data_name = snap_grp.listnames()[0]
                data_grp = snap_grp.require_group(data_name)
            else:
                data_name = fobj.keys()[0]
                data_grp = fobj.require_group(data_name)
            data = pickle.loads(data_grp.attrs['Class'])()
            for (k, v) in data_grp.items():
                obj = pickle.loads(v.attrs['Class'])()
                obj.set_data(v[:])
                data.set_members(obj)
        if isinstance(snap_name, str):
            return (data, snap_time)
        return data


########## end of file ##########
