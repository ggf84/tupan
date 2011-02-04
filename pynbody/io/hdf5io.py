#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import h5py

from pynbody import selftimer


class HDF5IO():
    """

    """
    def __init__(self):
        pass

    def read_snapshot(self):
        pass

    @selftimer
    def write_snapshot(self, data,
                       group_name='snapshot',
                       dset_name='wline',
                       file_name='output'):
        """

        """
        with h5py.File(file_name+'.hdf5', 'w') as fobj:
            snap = fobj.create_group(group_name)
            for (k, v) in data.items():
                subgrp = snap.create_group(k)
                if len(v) > 0:
                    subgrp.attrs['dtype'] = str(v.array.dtype)
                    wl = subgrp.create_dataset(dset_name, (len(v),1),
                                               dtype=v.array.dtype,
                                               maxshape=(None,None),
                                               chunks=True,
                                               compression='gzip',
                                               shuffle=True)
                    wl[:,0] = v.get_data()


########## end of file ##########
