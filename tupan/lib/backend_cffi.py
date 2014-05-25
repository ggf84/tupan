# -*- coding: utf-8 -*-
#

"""This module implements the CFFI backend to call C-extensions.

"""


import os
import cffi
import ctypes
import logging
from functools import partial
from collections import namedtuple
from .utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")


@timings
def make_lib(fpwidth):
    """

    """
    cint = "int" if fpwidth == "fp32" else "long"
    creal = "float" if fpwidth == 'fp32' else "double"
    LOGGER.debug("Building/Loading %s C extension module.", fpwidth)

    fnames = ("phi_kernel.c",
              "acc_kernel.c",
              "acc_jerk_kernel.c",
              "snap_crackle_kernel.c",
              "tstep_kernel.c",
              "pnacc_kernel.c",
              "nreg_kernels.c",
              "sakura_kernel.c",
              "kepler_solver_kernel.c", )

    src = []
    with open(os.path.join(PATH, "libtupan.h"), "r") as fobj:
        src.append("typedef {} INT;".format(cint))
        src.append("typedef unsigned {} UINT;".format(cint))
        src.append("typedef {} REAL;".format(creal))
        src.append(fobj.read())
    source = "\n".join(src)

    ffi = cffi.FFI()

    ffi.cdef(source)

    define_macros = []
    if fpwidth == "fp64":
        define_macros.append(("CONFIG_USE_DOUBLE", 1))

    from ..config import CACHE_DIR
    clib = ffi.verify(
        """
        #include "common.h"
        #include "libtupan.h"
        """,
        tmpdir=CACHE_DIR,
        define_macros=define_macros,
        include_dirs=[PATH],
        libraries=["m"],
        extra_compile_args=["-O3", "-std=c99"],
        sources=[os.path.join(PATH, fname) for fname in fnames],
    )

    LOGGER.debug("C extension module loaded: "
                 "(U)INT is (u)%s, REAL is %s.",
                 cint, creal)
    return ffi, clib


FFI = {}
LIB = {}
FFI['fp32'], LIB['fp32'] = make_lib('fp32')
FFI['fp64'], LIB['fp64'] = make_lib('fp64')


@bind_all(timings)
class CKernel(object):

    def __init__(self, fpwidth, name):
        self.kernel = getattr(LIB[fpwidth], name)
        self.argtypes = None
        self._args = None

        ffi = FFI[fpwidth]

#        from_buffer = ctypes.c_char.from_buffer
#        from_buffer = ctypes.POINTER(ctypes.c_char).from_buffer
        from_buffer = (ctypes.c_char * 0).from_buffer

        addressof = ctypes.addressof
        icast = partial(ffi.cast, "INT *")
        uicast = partial(ffi.cast, "UINT *")
        rcast = partial(ffi.cast, "REAL *")

        types = namedtuple("Types", ["c_int", "c_int_p",
                                     "c_uint", "c_uint_p",
                                     "c_real", "c_real_p"])
        self.cty = types(
            c_int=lambda x: x,
            c_int_p=lambda x: icast(addressof(from_buffer(x))),
            c_uint=lambda x: x,
            c_uint_p=lambda x: uicast(addressof(from_buffer(x))),
            c_real=lambda x: x,
            c_real_p=lambda x: rcast(addressof(from_buffer(x))),
            )

    def set_gsize(self, ni, nj):
        pass

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        argtypes = self.argtypes
        self._args = [argtype(arg) for (arg, argtype) in zip(args, argtypes)]

    def map_buffers(self, **kwargs):
        return kwargs['outargs']

    def run(self):
        self.kernel(*self.args)


# -- End of File --
