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

    def __init__(self, fname, output_format=None):
        self.fname = fname
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
        plt.plot(a1.pos[:,0], a1.pos[:,1], label='fixed-dt sampling')
        plt.plot(a.pos[:,0], a.pos[:,1], label='PBaSS: l=8')
        plt.legend(loc='best', shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()


        x = np.linspace(0,25,100000)
        k = 0
#        f1 = interpolate.UnivariateSpline(a.time, a.pos[:,k], s=0, k=1)
#        f2 = interpolate.UnivariateSpline(a.time, a.pos[:,k], s=0, k=2)
        f3 = interpolate.UnivariateSpline(a.time, a.pos[:,k], s=0, k=3)
#        f4 = interpolate.UnivariateSpline(a.time, a.pos[:,k], s=0, k=4)
#        f5 = interpolate.UnivariateSpline(a.time, a.pos[:,k], s=0, k=5)
#        y1 = f1(x)
#        y2 = f2(x)
        y3 = f3(x)
#        y4 = f4(x)
#        y5 = f5(x)
        plt.plot(a0.time, a0.pos[:,k], label='PBaSS: l=1')
        plt.plot(a1.time, a1.pos[:,k], label='fixed-dt sampling')
        plt.plot(a.time, a.pos[:,k], label='PBass: l=8')
#        plt.plot(x, y1, label='f1')
#        plt.plot(x, y2, label='f2')
        plt.plot(x, y3, label='3rd order interp. func.')
#        plt.plot(x, y4, label='f4')
#        plt.plot(x, y5, label='f5')
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
                                f = interpolate.UnivariateSpline(time, attr[:,k], s=0, k=3)
                                for t in times:
                                    getattr(getattr(snaps[t], key), name)[i,k] = f(t)
                        else:
                            f = interpolate.UnivariateSpline(time, attr[:], s=0, k=3)
                            for t in times:
                                getattr(getattr(snaps[t], key), name)[i] = f(t)

        return snaps.values()






########## end of file ##########
