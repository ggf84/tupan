#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import pickle
import h5py

from pynbody.lib.decorators import selftimer


class HDF5IO(object):
    """

    """
    def __init__(self, fname, fmode='a'):
        self.fname = fname
        self.fmode = fmode
        self.slice_size = 1


    def write_restart_file(self):
        pass

    def read_restart_file(self):
        pass


    @selftimer
    def write_snapshot(self, data,
                       group_name='snap',
                       group_id=None,
                       dset_name='dset',
                       slice_id=0):
        """

        """
        if group_id is not None:
            group_name += '_'+str(group_id).zfill(4)
        with h5py.File(self.fname, self.fmode) as fobj:
            snap = fobj.require_group(group_name)
            snap.attrs['snap_class'] = pickle.dumps(data.__class__)
            for (k, v) in data.items():
                subgrp = snap.require_group(k)
                if v:
                    subgrp.attrs['num_of_objs'] = pickle.dumps(len(v))
                    subgrp.attrs['obj_dtype'] = pickle.dumps(v._dtype)
                    subgrp.attrs['obj_class'] = pickle.dumps(v.__class__)
                    dset = subgrp.require_dataset(dset_name,
                                                  (len(v),self.slice_size),
                                                  dtype=v._dtype,
                                                  maxshape=(None,None),
                                                  chunks=True,
                                                  compression='gzip',
                                                  shuffle=True)
                    dset[:,slice_id] = v.get_data()


    @selftimer
    def read_snapshot(self,
                      group_name='snap',
                      group_id=None,
                      dset_name='dset',
                      slice_id=0):
        """

        """
        if group_id is not None:
            group_name += '_'+str(group_id).zfill(4)
        with h5py.File(self.fname, 'r') as fobj:
            snap = fobj.require_group(group_name)
            snap_class = pickle.loads(snap.attrs['snap_class'])
            data = snap_class()
            for (k, v) in snap.items():
                if v.attrs:
                    num_of_objs = pickle.loads(v.attrs['num_of_objs'])
                    obj_dtype = pickle.loads(v.attrs['obj_dtype'])
                    obj_class = pickle.loads(v.attrs['obj_class'])
                    dset = v.require_dataset(dset_name,
                                             (num_of_objs, self.slice_size),
                                             dtype=obj_dtype)
                    members = obj_class()
                    members.set_data(dset[:,slice_id])
                    data.set_members(members)
        return data


########## end of file ##########
