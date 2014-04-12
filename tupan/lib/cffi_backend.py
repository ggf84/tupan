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


logger = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")


def make_lib(prec):
    """

    """
    cint = "int" if prec == "float32" else "long"
    creal = "float" if prec == 'float32' else "double"
    logger.debug("Building/Loading %s C extension module.", prec)

    fnames = ("phi_kernel.c",
              "acc_kernel.c",
              "acc_jerk_kernel.c",
              "snap_crackle_kernel.c",
              "tstep_kernel.c",
              "pnacc_kernel.c",
              "nreg_kernels.c",
              "sakura_kernel.c",
              "kepler_solver_kernel.c",
              )

    s = []
    with open(os.path.join(PATH, "libtupan.h"), "r") as fobj:
        s.append("typedef {} INT;".format(cint))
        s.append("typedef unsigned {} UINT;".format(cint))
        s.append("typedef {} REAL;".format(creal))
        s.append(fobj.read())
    source = "\n".join(s)

    ffi = cffi.FFI()

    ffi.cdef(source)

    define_macros = []
    if prec == "float64":
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

    logger.debug("C extension module loaded: "
                 "(U)INT is (u)%s, REAL is %s.",
                 cint, creal)
    return ffi, clib


FFI = {}
LIB = {}
FFI['float32'], LIB['float32'] = make_lib('float32')
FFI['float64'], LIB['float64'] = make_lib('float64')


class CKernel(object):

    def __init__(self, prec, name):
        self.kernel = getattr(LIB[prec], name)
        self._args = None
        self._argtypes = None

        ffi = FFI[prec]

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
        self.cty = types(c_int=lambda x: x,
                         c_int_p=lambda x: icast(addressof(from_buffer(x))),
                         c_uint=lambda x: x,
                         c_uint_p=lambda x: uicast(addressof(from_buffer(x))),
                         c_real=lambda x: x,
                         c_real_p=lambda x: rcast(addressof(from_buffer(x))),
                         )

    def set_gsize(self, ni, nj):
        pass

    @property
    def argtypes(self):
        return self._argtypes

    @argtypes.setter
    def argtypes(self, types):
        self._argtypes = types

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
