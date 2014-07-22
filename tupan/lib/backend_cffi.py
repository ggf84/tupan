# -*- coding: utf-8 -*-
#

"""This module implements the CFFI backend to call C-extensions.

"""


import os
import cffi
import ctypes
import logging
from tupan.config import options
from .utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'src')


@bind_all(timings)
class CDriver(object):
    """

    """
    def __init__(self, fpwidth=options.fpwidth):
        self.fpwidth = fpwidth

        self._make_lib()

    def _make_lib(self):
        cint = 'int' if self.fpwidth == 'fp32' else 'long'
        creal = 'float' if self.fpwidth == 'fp32' else 'double'
        LOGGER.debug("Building '%s' C extension module.", self.fpwidth)

        fnames = ('phi_kernel.c',
                  'acc_kernel.c',
                  'acc_jerk_kernel.c',
                  'snap_crackle_kernel.c',
                  'tstep_kernel.c',
                  'pnacc_kernel.c',
                  'nreg_kernels.c',
                  'sakura_kernel.c',
                  'kepler_solver_kernel.c', )

        src = []
        with open(os.path.join(PATH, 'libtupan.h'), 'r') as fobj:
            src.append('typedef {} INT;'.format(cint))
            src.append('typedef unsigned {} UINT;'.format(cint))
            src.append('typedef {} REAL;'.format(creal))
            src.append(fobj.read())
        source = '\n'.join(src)

        self.ffi = cffi.FFI()

        self.ffi.cdef(source)

        define_macros = []
        if self.fpwidth == 'fp64':
            define_macros.append(('CONFIG_USE_DOUBLE', 1))

        from ..config import get_cache_dir
        self.lib = self.ffi.verify(
            """
            #include "common.h"
            #include "libtupan.h"
            """,
            tmpdir=get_cache_dir(),
            define_macros=define_macros,
            include_dirs=[PATH],
            libraries=['m'],
            extra_compile_args=['-O3', '-std=c99'],
            sources=[os.path.join(PATH, fname) for fname in fnames],
        )

        LOGGER.debug('C extension module loaded: '
                     '(U)INT is (u)%s, REAL is %s.',
                     cint, creal)

#        from_buffer = ctypes.c_char.from_buffer
#        from_buffer = ctypes.POINTER(ctypes.c_char).from_buffer
        from_buffer = (ctypes.c_char * 0).from_buffer
        addressof = ctypes.addressof
        cast = self.ffi.cast

        self.c_int = lambda x: x
        self.c_uint = lambda x: x
        self.c_real = lambda x: x
        self.iptr = lambda x: cast('void *', addressof(from_buffer(x)))
        self.optr = lambda x: cast('void *', addressof(from_buffer(x)))

    def get_kernel(self, name):
        LOGGER.debug("Using '%s' function from 'C' backend.", name)
        kernel = getattr(self.lib, name)
        return CKernel(kernel)


drv = CDriver()


@bind_all(timings)
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

    def set_args(self, inpargs, outargs):
        bufs = []

        if self.inptypes is None:
            types = []
            for arg in inpargs:
                if isinstance(arg, int):
                    types.append(drv.c_int if arg < 0 else drv.c_uint)
                elif isinstance(arg, float):
                    types.append(drv.c_real)
                else:
                    types.append(drv.iptr)
            self.inptypes = types

        if self.outtypes is None:
            self.outtypes = [drv.optr for _ in outargs]

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
