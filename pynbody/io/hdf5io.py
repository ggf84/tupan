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
    def __init__(self, fname):
        self.hdf5filename = fname
        self.slice_size = 1

    @selftimer
    def write_snapshot(self, data,
                       group_name='snapshot',
                       dset_name='dset',
                       slice_id=0):
        """

        """
        with h5py.File(self.hdf5filename, 'a') as fobj:
            snap = fobj.require_group(group_name)
            snap.attrs['snap_class'] = pickle.dumps(data.__class__)
            for (k, v) in data.items():
                subgrp = snap.require_group(k)
                if len(v) > 0:
                    subgrp.attrs['num_of_particles'] = pickle.dumps(len(v))
                    subgrp.attrs['particle_dtype'] = pickle.dumps(v.array.dtype)
                    subgrp.attrs['particle_class'] = pickle.dumps(v.__class__)
                    dset = subgrp.require_dataset(dset_name,
                                                  (len(v),self.slice_size),
                                                  dtype=v.array.dtype,
                                                  maxshape=(None,None),
                                                  chunks=True,
                                                  compression='gzip',
                                                  shuffle=True)
                    dset[:,slice_id] = v.get_data()


    @selftimer
    def read_snapshot(self, group_name='snapshot',
                      dset_name='dset',
                      slice_id=0):
        """

        """
        with h5py.File(self.hdf5filename, 'r') as fobj:
            snap = fobj.require_group(group_name)
            snap_class = pickle.loads(snap.attrs['snap_class'])
            data = snap_class()
            for (k, v) in snap.items():
                if v.attrs.keys():
                    num_of_particles = pickle.loads(v.attrs['num_of_particles'])
                    particle_dtype = pickle.loads(v.attrs['particle_dtype'])
                    particle_class = pickle.loads(v.attrs['particle_class'])
                    dset = v.require_dataset(dset_name,
                                             (num_of_particles, self.slice_size),
                                             dtype=particle_dtype)
                    members = particle_class()
                    members.fromlist(dset[:,slice_id])
                    data.set_members(members)
        return data



########## end of file ##########
