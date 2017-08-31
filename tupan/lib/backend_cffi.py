# -*- coding: utf-8 -*-
#

"""
This module implements the CFFI backend to call C-extensions.
"""

import importlib
from ..config import cli


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

    def make_struct(self, name, **kwargs):
        return {'struct': drv.ffi.new(name+' *', kwargs)}

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
            cast = drv.ffi.cast
            from_buffer = drv.ffi.from_buffer
            for arg in inpargs:
                if isinstance(arg, int):
                    types.append(lambda x: x)
                elif isinstance(arg, float):
                    types.append(lambda x: x)
                elif isinstance(arg, dict):
                    types.append(lambda x: x['struct'][0])
                else:
                    types.append(lambda x: cast('real_t *', from_buffer(x)))
            self.inptypes = types
            self.outtypes = [lambda x: cast('real_t *', from_buffer(x))
                             for _ in outargs]
            return CKernel.set_args(self, inpargs, outargs)

    def map_buffers(self, obufs):
        pass

    def run(self, ibufs, obufs):
        bufs = (buf for (arg, buf) in ibufs+obufs)
        self.kernel(*bufs)


# -- End of File --
