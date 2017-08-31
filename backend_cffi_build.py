#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Implements CFFI builder functions for C-extensions.
"""


import os
from cffi import FFI


CWD = os.path.dirname(__file__)
PATH = os.path.join(CWD, 'tupan', 'lib', 'src')


FILE_NAMES = [
    'phi_kernel.cpp',
    'acc_kernel.cpp',
    'acc_jrk_kernel.cpp',
    'snp_crk_kernel.cpp',
    'tstep_kernel.cpp',
    'pnacc_kernel.cpp',
    'sakura_kernel.cpp',
    'kepler_solver_kernel.cpp',
]


COMPILER_FLAGS = [
    '-O2',
    '-flto',
    '-fopenmp',
    '-ffast-math',
    '-funroll-loops',
    '-std=c++14',
#    '-fopt-info-vec',              # gcc
#    '-Rpass=loop-vectorize',       # clang
]


def ffibuilder(fp):
    ffi = FFI()

    define_macros = {}
    define_macros['SIMD'] = 1
#    define_macros['CONFIG_DEBUG'] = 1
    if fp == 64:
        define_macros['CONFIG_USE_DOUBLE'] = 1

    ffi.set_source(
        "tupan.lib.libtupan"+str(fp),
        """
        #include "common.h"
        """,
        define_macros=list(define_macros.items()),
        include_dirs=[PATH],
        libraries=['m'],
        extra_compile_args=COMPILER_FLAGS,
        extra_link_args=COMPILER_FLAGS,
        source_extension='.cpp',
        sources=[os.path.join(PATH, fname) for fname in FILE_NAMES],
    )

    src = []
    with open(os.path.join(PATH, 'libtupan.h'), 'r') as fobj:
        src.append('typedef int... int_t;')
        src.append('typedef unsigned int... uint_t;')
        src.append('typedef float... real_t;')
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
            '''
        )
        src.append(fobj.read())
    source = '\n'.join(src)
    ffi.cdef(source)
    return ffi


def ffibuilder32():
    return ffibuilder(32)


def ffibuilder64():
    return ffibuilder(64)


if __name__ == "__main__":
    ffibuilder(32).compile(verbose=True)
    ffibuilder(64).compile(verbose=True)


# -- End of File --
