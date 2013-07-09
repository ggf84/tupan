# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import os
import sys
import cffi


DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")


def wrap_lib(fptype):
    sources = ("smoothing.c",
               "universal_kepler_solver.c",
               #
               "phi_kernel.c",
               "acc_kernel.c",
               "acc_jerk_kernel.c",
               "tstep_kernel.c",
               "pnacc_kernel.c",
               "nreg_kernels.c",
               "sakura_kernel.c",
               )

    with open(os.path.join(PATH, "libtupan.h"), "r") as fobj:
        source = "typedef {} REAL;".format(fptype) + fobj.read()

    ffi = cffi.FFI()

    ffi.cdef(source)

    define_macros = []
    modulename = "libTupanSPcffi"
    if fptype == "double":
        define_macros.append(("DOUBLE", 1))
        modulename = modulename.replace("SP", "DP")

    clib = ffi.verify(
        """
        #include "common.h"
        #include "libtupan.h"
        """,
        define_macros=define_macros,
        include_dirs=[PATH],
        libraries=["m"],
        extra_compile_args=["-O3"],
        sources=[os.path.join(PATH, src) for src in sources],
        force_generic_engine=hasattr(
            sys, "_force_generic_engine_"),
        modulename=modulename,
    )
    return ffi, clib


def get_extensions():
    ffi_sp, clib_sp = wrap_lib("float")
    ffi_dp, clib_dp = wrap_lib("double")
    return [ffi_sp.verifier.get_extension(), ffi_dp.verifier.get_extension()]


########## end of file ##########
