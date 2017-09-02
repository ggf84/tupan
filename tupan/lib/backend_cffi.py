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

    def map_buffers(self, oargs, obufs):
        pass

    def run(self, bufs):
        self.kernel(*bufs)


# -- End of File --
