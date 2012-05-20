#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import pickle
import h5py
from ..lib.utils.timing import decallmethods, timings


__all__ = ['HDF5IO']


@decallmethods(timings)
class HDF5IO(object):
    """

    """
    def __init__(self, fname):
        self.fname = fname


    def dump(self, particles, fmode='a'):
        """

        """
        with h5py.File(self.fname, fmode) as fobj:
            group_name = particles.__class__.__name__.lower()
            group = fobj.require_group(group_name)
            group.attrs['Class'] = pickle.dumps(particles.__class__)
            for (k, v) in particles.items():
                if v.n:
                    dset_name = v.__class__.__name__.lower()
                    dset_length = 0
                    if k in group:
                        dset_length = len(group[k])
                    state = v.get_state()   # XXX:
                    dset = group.require_dataset(dset_name,
                                                 (dset_length,),
#                                                 dtype=v.dtype,
                                                 dtype=state.dtype,
                                                 maxshape=(None,),
                                                 chunks=True,
                                                 compression='gzip',
                                                 shuffle=True,
                                                )
                    dset.attrs['Class'] = pickle.dumps(v.__class__)
                    olen = len(dset)
                    dset.resize((olen+len(v),))
                    nlen = len(dset)
#                    dset[olen:nlen] = v.data
                    dset[olen:nlen] = state


    def load(self):
        """

        """
        with h5py.File(self.fname, 'r') as fobj:
            group_name = fobj.keys()[0]
            group = fobj.require_group(group_name)
            particles = pickle.loads(group.attrs['Class'])()
            for (k, v) in group.items():
                obj = pickle.loads(v.attrs['Class'])()
#                obj.data = v[:]
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
