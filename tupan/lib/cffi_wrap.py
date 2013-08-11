# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import os
import cffi


DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")


def wrap_lib(fptype):
    files = ("smoothing.c",
             "universal_kepler_solver.c",
             #
             "phi_kernel.c",
             "acc_kernel.c",
             "acc_jerk_kernel.c",
             "snap_crackle_kernel.c",
             "tstep_kernel.c",
             "pnacc_kernel.c",
             "nreg_kernels.c",
             "sakura_kernel.c",
             )

    s = []
    with open(os.path.join(PATH, "libtupan.h"), "r") as fobj:
        s.append("typedef unsigned int UINT;")
        s.append("typedef {} REAL;".format(fptype))
        s.append(fobj.read())
    source = "\n".join(s)

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
        sources=[os.path.join(PATH, file) for file in files],
        modulename=modulename,
    )
    return ffi, clib


def get_extensions():
    ffi_sp, clib_sp = wrap_lib("float")
    ffi_dp, clib_dp = wrap_lib("double")
    return [ffi_sp.verifier.get_extension(), ffi_dp.verifier.get_extension()]


########## end of file ##########
