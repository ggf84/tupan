#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from . import hdf5io
from . import psdfio


class IO(object):
    """

    """

    PROVIDED_FORMATS = ['hdf5', 'psdf']

    def __init__(self, fname, fmode, output_format=None):
        self.fname = fname
        self.fmode = fmode
        self.output_format = output_format


    def setup(self, *args, **kwargs):
        fname = self.fname
        if self.output_format == "psdf":
            if not fname.endswith(".psdf"):
                fname += ".psdf"
            self.io_obj = psdfio.PSDFIO(fname).setup(*args, **kwargs)
        elif self.output_format == "hdf5":
            if not fname.endswith(".hdf5"):
                fname += ".hdf5"
            self.io_obj = hdf5io.HDF5IO(fname).setup(*args, **kwargs)
        else:
            raise NotImplementedError('Unknown format: {}. '
                                      'Choose from: {}'.format(self.output_format,
                                       self.PROVIDED_FORMATS))
        return self


    def append(self, *args, **kwargs):
        self.io_obj.append(*args, **kwargs)


    def flush(self, *args, **kwargs):
        self.io_obj.flush(*args, **kwargs)


    def close(self, *args, **kwargs):
        self.io_obj.close(*args, **kwargs)


    def dumpper(self, *args, **kwargs):
        self.io_obj.dumpper(*args, **kwargs)


    def dump(self, *args, **kwargs):
        fname = self.fname
        if self.output_format == "psdf":
            if not fname.endswith(".psdf"):
                fname += ".psdf"
            psdfio.PSDFIO(fname).dump(*args, **kwargs)
        elif self.output_format == "hdf5":
            if not fname.endswith(".hdf5"):
                fname += ".hdf5"
            hdf5io.HDF5IO(fname).dump(*args, **kwargs)
        else:
            raise NotImplementedError('Unknown format: {}. '
                                      'Choose from: {}'.format(self.output_format,
                                       self.PROVIDED_FORMATS))


    def dump_snapshot(self, *args, **kwargs):
        fname = self.fname
        fmode = self.fmode
        if self.output_format == "psdf":
            if not fname.endswith(".psdf"):
                fname += ".psdf"
            psdfio.PSDFIO(fname, fmode).dump_snapshot(*args, **kwargs)
        elif self.output_format == "hdf5":
            if not fname.endswith(".hdf5"):
                fname += ".hdf5"
            hdf5io.HDF5IO(fname, fmode).dump_snapshot(*args, **kwargs)
        else:
            raise NotImplementedError('Unknown format: {}. '
                                      'Choose from: {}'.format(self.output_format,
                                       self.PROVIDED_FORMATS))


    def load(self, *args, **kwargs):
        import os
        import sys
        import logging
        logger = logging.getLogger(__name__)
        from warnings import warn
        fname = self.fname
        if not os.path.exists(fname):
            warn("No such file or directory: '{}'".format(fname), stacklevel=2)
            sys.exit()
        loaders = (psdfio.PSDFIO, hdf5io.HDF5IO,)
        for loader in loaders:
            try:
                return loader(fname).load(*args, **kwargs)
            except Exception as exc:
                pass
        logger.exception(str(exc))
        raise ValueError("File not in a supported format.")


    def to_psdf(self):
        fname = self.fname
        loaders = (hdf5io.HDF5IO,)
        for loader in loaders:
            try:
                return loader(fname).to_psdf()
            except Exception:
                pass
        raise ValueError('This file is already in \'psdf\' format!')


    def to_hdf5(self):
        fname = self.fname
        loaders = (psdfio.PSDFIO,)
        for loader in loaders:
            try:
                return loader(fname).to_hdf5()
            except Exception:
                pass
        raise ValueError('This file is already in \'hdf5\' format!')





    def take_time_slices(self, times):
        import numpy as np
        from scipy import interpolate
        from collections import OrderedDict


        #######################################################################
        import matplotlib.pyplot as plt
        from pynbody.io import IO
        p0 = IO('snapshots0.hdf5').load()
        p1 = IO('snapshots1.hdf5').load()
        p = IO('snapshots.hdf5').load()

        index = p[p.nstep == p.nstep.max()].id[0]
        a0 = p0[p0.id == index]
        a1 = p1[p1.id == index]
        a = p[p.id == index]

        plt.plot(a0.pos[:,0], a0.pos[:,1], label='PBaSS: l=1')
        plt.plot(a1.pos[:,0], a1.pos[:,1], label='PBaSS: l=8')
        plt.plot(a.pos[:,0], a.pos[:,1], label='PBaSS: l=64')
        plt.legend(loc='best', shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()

        axis = 0
        x = np.linspace(0,25,1000000)
        f = interpolate.UnivariateSpline(a1.time, a1.pos[:,axis], s=0, k=2)
        plt.plot(a0.time, a0.pos[:,axis], label='PBaSS: l=1')
        plt.plot(a1.time, a1.pos[:,axis], label='PBaSS: l=8')
        plt.plot(a.time, a.pos[:,axis], label='PBass: l=64')
        plt.plot(x, f(x), label='interp. function')
        plt.legend(loc='best', shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()
        #######################################################################


        p = self.load()

        snaps = OrderedDict()
        for t in times:
            snaps[t] = type(p)()

        for key, obj in p.items:
            if obj.n:
                n = int(obj.id.max())+1

                snap = type(obj)(n)
                for t in times:
                    snaps[t].append(snap)

                for i in xrange(n):
                    stream = obj[obj.id == i]
                    time = stream.time
                    for name in obj.names:
                        attr = getattr(stream, name)
                        if attr.ndim > 1:
                            for k in xrange(attr.shape[1]):
                                f = interpolate.UnivariateSpline(time, attr[:,k], s=0, k=2)
                                for t in times:
                                    getattr(getattr(snaps[t], key), name)[i,k] = f(t)
                        else:
                            f = interpolate.UnivariateSpline(time, attr[:], s=0, k=2)
                            for t in times:
                                getattr(getattr(snaps[t], key), name)[i] = f(t)

        return snaps.values()






########## end of file ##########
