# -*- coding: utf-8 -*-
#

"""
This module implements the CFFI backend to call C-extensions.
"""

import os
import cffi
import logging
from ..config import options, get_cache_dir
from .utils.ctype import Ctype
from .utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'src')


@bind_all(timings)
class CDriver(object):
    """

    """
    def __init__(self, fpwidth=options.fpwidth):
        LOGGER.debug("Building '%s' C extension module.", fpwidth)

        fnames = (
            'phi_kernel.cpp',
            'acc_kernel.cpp',
            'acc_jrk_kernel.cpp',
            'snp_crk_kernel.cpp',
            'tstep_kernel.cpp',
            'pnacc_kernel.cpp',
            'nreg_kernels.cpp',
            'sakura_kernel.cpp',
            'kepler_solver_kernel.cpp',
        )

        src = []
        with open(os.path.join(PATH, 'libtupan.h'), 'r') as fobj:
            src.append('typedef {} int_t;'.format(Ctype.c_int))
            src.append('typedef {} uint_t;'.format(Ctype.c_uint))
            src.append('typedef {} real_t;'.format(Ctype.c_real))
            src.append(
                '''
                typedef struct clight_struct {
                    real_t inv1;
                    real_t inv2;
                    real_t inv3;
                    real_t inv4;
                    real_t inv5;
                    real_t inv6;
                    real_t inv7;
                    uint_t order;
                } CLIGHT;
                ''')
            src.append(fobj.read())
        source = '\n'.join(src)

        self.ffi = cffi.FFI()

        self.ffi.cdef(source)

        define_macros = {}
        define_macros['SIMD'] = 1
        if fpwidth == 'fp64':
            define_macros['CONFIG_USE_DOUBLE'] = 1

        compiler_flags = [
            '-O3',
            '-std=c++14',
            '-march=native',
            '-fpermissive',
            '-fopenmp',
        ]

        self.lib = self.ffi.verify(
            """
            #include "common.h"
            """,
            tmpdir=get_cache_dir(fpwidth),
            define_macros=list(define_macros.items()),
            include_dirs=[PATH],
            libraries=['m'],
            extra_compile_args=compiler_flags,
            extra_link_args=compiler_flags,
            source_extension='.cpp',
            sources=[os.path.join(PATH, fname) for fname in fnames],
        )

        LOGGER.debug('C extension module loaded.')

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
