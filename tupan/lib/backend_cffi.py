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
        cast = drv.ffi.cast
        from_buffer = drv.ffi.from_buffer
        if hasattr(self, 'both'):
            self.triangle = getattr(drv.lib, name+'_triangle')
            self.rectangle = getattr(drv.lib, name+'_rectangle')
        else:
            self.run = getattr(drv.lib, name)
        self.to_int = Ctype.int_t
        self.to_real = Ctype.real_t
        self.to_buffer = lambda x: cast('real_t *', from_buffer(x))
        self.map_bufs = lambda x, y: None
        self.map_buffers = lambda x, y: None


# -- End of File --
