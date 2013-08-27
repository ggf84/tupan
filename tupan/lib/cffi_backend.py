# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import os
import sys
import cffi
import ctypes
import logging
import getpass
import tempfile
from functools import partial
from collections import namedtuple


logger = logging.getLogger(__name__)


DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")

TMPDIR = os.path.join(tempfile.gettempdir(),
                      "tupan-cffi-cache-uid{0}-py{1}".format(
                          getpass.getuser(),
                          ".".join(str(i) for i in sys.version_info)))


def get_lib(env):
    """

    """
    logger.debug("Building/Loading %s precision C extension module...",
                 env.prec)

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
        s.append("typedef {} REAL;".format(env.fptype))
        s.append(fobj.read())
    source = "\n".join(s)

    ffi = cffi.FFI()

    ffi.cdef(source)

    define_macros = []
    modulename = "libTupanSPcffi"
    if env.fptype == "double":
        define_macros.append(("DOUBLE", 1))
        modulename = modulename.replace("SP", "DP")

    clib = ffi.verify(
        """
        #include "common.h"
        #include "libtupan.h"
        """,
        tmpdir=TMPDIR,
        define_macros=define_macros,
        include_dirs=[PATH],
        libraries=["m"],
        extra_compile_args=["-O3"],
        sources=[os.path.join(PATH, file) for file in files],
        modulename=modulename,
    )

    logger.debug("done.")
    return clib


class CKernel(object):

    def __init__(self, env, ffi, kernel):
        self.env = env
        self.kernel = kernel

        from_buffer = ctypes.c_char.from_buffer
        addressof = ctypes.addressof
        icast = partial(ffi.cast, "UINT *")
        rcast = partial(ffi.cast, "REAL *")

        types = namedtuple("Types", ["c_uint", "c_uint_p",
                                     "c_real", "c_real_p"])
        self.cty = types(c_uint=lambda x: x,
                         c_uint_p=lambda x: icast(addressof(from_buffer(x))),
                         c_real=lambda x: x,
                         c_real_p=lambda x: rcast(addressof(from_buffer(x))),
                         )

    def allocate_local_memory(self, numbufs, dtype):
        return []

    def set_args(self, args, start=0):
        self.args = args

    def map_buffers(self, arrays, buffers):
        return arrays

    def run(self):
        self.kernel(*self.args)


########## end of file ##########
