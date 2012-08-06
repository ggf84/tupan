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


########## end of file ##########
