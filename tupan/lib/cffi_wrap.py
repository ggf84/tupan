
from __future__ import print_function
import os
import sys
import tempfile
import cffi


DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")

TMPDIR = os.path.join(tempfile.gettempdir(), "__pycache__")


def wrap_lib(fptype):

    with open(os.path.join(PATH, "libtupan.h"), "r") as fobj:
        src = fobj.read()
        source = "typedef {} REAL, *pREAL;".format(fptype) + src

    ffi = cffi.FFI()

    ffi.cdef(source)

    define_macros = []
    modulename = "libTupanSPcffi"
    if fptype == "double":
        define_macros.append(("DOUBLE", 1))
        modulename = modulename.replace("SP", "DP")

    clib = ffi.verify("""
    #include<common.h>
    #include<libtupan.h>
    """,
                      tmpdir=TMPDIR,
                      define_macros=define_macros,
                      include_dirs=[PATH],
                      libraries=['m'],
                      sources=[os.path.join(PATH, 'libtupan.c')],
                      force_generic_engine=hasattr(sys, '_force_generic_engine_'),
                      modulename=modulename,
                     )
    return ffi, clib

try:
    ffi_sp, clib_sp = wrap_lib("float")
    ffi_dp, clib_dp = wrap_lib("double")
except Exception as exc:
    print(str(exc), file=sys.stderr)
    ffi_sp, clib_sp = None, None
    ffi_dp, clib_dp = None, None


def get_extensions():
    return [ffi_sp.verifier.get_extension(), ffi_dp.verifier.get_extension()]

