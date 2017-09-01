# -*- coding: utf-8 -*-
#

"""
This module implements the CFFI backend to call C-extensions.
"""

import importlib
from ..config import cli, Ctype


libtupan = importlib.import_module(
    '.libtupan'+str(cli.fp),
    package=__package__
)


class CDriver(object):
    """

    """
    def __init__(self, libtupan):
        self.lib = libtupan.lib
        self.ffi = libtupan.ffi


drv = CDriver(libtupan)


class CKernel(object):
    """

    """
    def __init__(self, name):
        self.kernel = getattr(libtupan.lib, name)

        cast = drv.ffi.cast
        from_buffer = drv.ffi.from_buffer

        self.to_int = Ctype.int_t
        self.to_real = Ctype.real_t
        self.to_buffer = lambda x: cast('real_t *', from_buffer(x))

    def set_args(self, inpargs, outargs):
        try:
            ibufs = []
            obufs = []
            # set inpargs
            for (i, argtype) in enumerate(self.inptypes):
                arg = inpargs[i]
                buf = argtype(arg)
                ibufs.append((arg, buf))
            # set outargs
            for (i, argtype) in enumerate(self.outtypes):
                arg = outargs[i]
                arg[...] = 0
                buf = argtype(arg)
                obufs.append((arg, buf))
            return ibufs, obufs
        except AttributeError:
            types = []
            for arg in inpargs:
                if isinstance(arg, int):
                    types.append(self.to_int)
                elif isinstance(arg, float):
                    types.append(self.to_real)
                else:
                    types.append(self.to_buffer)
            self.inptypes = types
            self.outtypes = [self.to_buffer for _ in outargs]
            return CKernel.set_args(self, inpargs, outargs)

    def map_buffers(self, obufs):
        pass

    def run(self, ibufs, obufs):
        self.kernel(*map(lambda x: x[1], ibufs+obufs))


# -- End of File --
