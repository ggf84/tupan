# -*- coding: utf-8 -*-
#

"""
This module implements the CFFI backend to call C-extensions.
"""

import importlib
import numpy as np
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

        cast = self.ffi.cast
        from_buffer = self.ffi.from_buffer
        self.to_int = Ctype.int_t
        self.to_uint = Ctype.uint_t
        self.to_real = Ctype.real_t
        self.to_buf = lambda x: cast('void *', from_buffer(x))


drv = CDriver(libtupan)


class CKernel(object):
    """

    """
    def __init__(self, name):
        triangle = rectangle = name
        if 'kepler' not in name:
            triangle += '_triangle'
            rectangle += '_rectangle'
        self.triangle = getattr(drv.lib, triangle)
        self.rectangle = getattr(drv.lib, rectangle)
        self.to_int = drv.to_int
        self.to_uint = drv.to_uint
        self.to_real = drv.to_real
        self.to_ibuf = lambda obj: obj.buf
        self.to_obuf = lambda obj: obj.buf
        self.map_buf = lambda obj: None


def to_cbuf(ary):
    ptr = np.array(ary, copy=False, order='C')
    buf = drv.to_buf(ptr)
    return ptr, buf


# -- End of File --
