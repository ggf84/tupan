# -*- coding: utf-8 -*-
#

"""
This module implements the CFFI backend to call C-extensions.
"""

import importlib
from ..config import cli


libtupan = importlib.import_module(
    '.libtupan'+cli.fpwidth,
    package=__package__
)


class CDriver(object):
    """

    """
    def __init__(self, libtupan):
        self.lib = libtupan.lib
        self.ffi = libtupan.ffi

    def get_kernel(self, name):
        kernel = getattr(self.lib, name)
        return CKernel(kernel)


drv = CDriver(libtupan)


class CKernel(object):
    """

    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.inptypes = None
        self.outtypes = None
        self.iarg = {}
        self.ibuf = {}
        self.oarg = {}
        self.obuf = {}

    def make_struct(self, name, **kwargs):
        return {'struct': drv.ffi.new(name+' *', kwargs)}

    def set_args(self, inpargs, outargs):
        bufs = []

        if self.inptypes is None:
            types = []
            for arg in inpargs:
                if isinstance(arg, int):
                    types.append(lambda x: x)
                elif isinstance(arg, float):
                    types.append(lambda x: x)
                elif isinstance(arg, dict):
                    types.append(lambda x: x['struct'][0])
                else:
                    types.append(drv.ffi.from_buffer)
            self.inptypes = types

        if self.outtypes is None:
            self.outtypes = [drv.ffi.from_buffer for _ in outargs]

        # set inpargs
        for (i, argtype) in enumerate(self.inptypes):
            arg = inpargs[i]
            buf = argtype(arg)
            self.iarg[i] = arg
            self.ibuf[i] = buf
            bufs.append(buf)

        # set outargs
        for (i, argtype) in enumerate(self.outtypes):
            arg = outargs[i]
            buf = argtype(arg)
            self.oarg[i] = arg
            self.obuf[i] = buf
            bufs.append(buf)

        self.bufs = bufs

    def map_buffers(self):
        return list(self.oarg.values())

    def run(self):
        self.kernel(*self.bufs)


# -- End of File --
