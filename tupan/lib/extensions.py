#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import os
import sys
import logging
from functools import reduce
from collections import OrderedDict
import numpy as np
import pyopencl as cl
from .utils.timing import decallmethods, timings
from .utils.dtype import *


logger = logging.getLogger(__name__)
logging.basicConfig(filename="spam.log", filemode='w',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.DEBUG)


@decallmethods(timings)
class CLEnv(object):

    def __init__(self, fast_math=True):
        self.fast_math = fast_math
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)



@decallmethods(timings)
class CLModule(object):

    def __init__(self, env):
        self.env = env

        # read the kernel source code
        dirname = os.path.dirname(__file__)
        abspath = os.path.abspath(dirname)
        path = os.path.join(abspath, "src")
        src_file = "libtupan.cl"
        fname = os.path.join(path, src_file)
        with open(fname, 'r') as fobj:
            src = fobj.read()
        self.src = src
        self.path = path


    def build(self, junroll=2):
        prec = "double" if REAL is np.float64 else "single"
        logger.debug("Building %s precision CL extension module.", prec)

        # setting options
        options = " -I {path}".format(path=self.path)
        options += " -D JUNROLL={junroll}".format(junroll=junroll)
        if prec is "double":
            options += " -D DOUBLE"
        if self.env.fast_math:
            options += " -cl-fast-relaxed-math"

        # building program
        self.program = cl.Program(self.env.ctx, self.src).build(options=options)

        logger.debug("done.")
        return self


    def __getattr__(self, name):
        return CLKernel(self.env, getattr(self.program, name))



@decallmethods(timings)
class CLKernel(object):

    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel
        self.dev_buff = {}
        self._lsize = None
        self._lsize_max = None
        self._gsize = None


    @property
    def local_size(self):
        lsize = self._lsize
        return lsize if isinstance(lsize, tuple) else (lsize,)

    @local_size.setter
    def local_size(self, lsize):
        self._lsize = lsize
        self._lsize_max = lsize


    @property
    def global_size(self):
        gsize = self._gsize
        return gsize if isinstance(gsize, tuple) else (gsize,)

    @global_size.setter
    def global_size(self, ni):
        self._gsize = ((ni-1)//2 + 1) * 2


    def set_local_memory(self, i, arg):
        def foo(x, y): return x * y
#        size = np.dtype(REAL).itemsize * reduce(foo, self.local_size)
#        size = np.dtype(REAL).itemsize * reduce(foo, (self._lsize_max,))
        size = 8 * reduce(foo, (self._lsize_max,))
        arg = cl.LocalMemory(size * arg)
        self.kernel.set_arg(i, arg)


    def set_int(self, i, arg):
        arg = UINT(arg)
        self.kernel.set_arg(i, arg)


    def set_float(self, i, arg):
        arg = REAL(arg)
        self.kernel.set_arg(i, arg)


    def set_array(self, i, arr):
        memf = cl.mem_flags
        self.dev_buff[i] = cl.Buffer(self.env.ctx,
                                     memf.READ_WRITE | memf.COPY_HOST_PTR,
                                     hostbuf=arr)

        arg = self.dev_buff[i]
        self.kernel.set_arg(i, arg)


    def allocate_buffer(self, i, shape):
        memf = cl.mem_flags
        arr = np.zeros(shape, dtype=REAL)
        self.dev_buff[i] = cl.Buffer(self.env.ctx,
                                     memf.READ_WRITE | memf.USE_HOST_PTR,
                                     hostbuf=arr)
#                                     memf.READ_WRITE | memf.ALLOC_HOST_PTR,
#                                     size=arr.nbytes)
        arg = self.dev_buff[i]
        self.kernel.set_arg(i, arg)
        return arr


    def map_buffer(self, i, arr):
        mapf = cl.map_flags
        (pointer, ev) = cl.enqueue_map_buffer(self.env.queue, self.dev_buff[i],
                                              mapf.READ, 0, arr.shape,
                                              arr.dtype, "C")
        ev.wait()

#        cl.enqueue_copy(self.env.queue, arr, self.dev_buff[i])


    def run(self):
        cl.enqueue_nd_range_kernel(self.env.queue,
                                   self.kernel,
                                   self.global_size,
                                   None,#self.local_size,
                                  ).wait()



@decallmethods(timings)
class CEnv(object):

    def __init__(self, fast_math=True):
        self.fast_math = fast_math



@decallmethods(timings)
class CModule(object):

    def __init__(self, env):
        self.env = env


    def build(self):
        prec = "double" if REAL is np.float64 else "single"
        logger.debug("Building %s precision C extension module.", prec)

        try:
            if prec is "double":
                from .cffi_wrap import clib_dp as program
                from .cffi_wrap import ffi_dp as ffi
            else:
                from .cffi_wrap import clib_sp as program
                from .cffi_wrap import ffi_sp as ffi
            self.program = program
            self.ffi = ffi
        except Exception as exc:
            print(str(exc), file=sys.stderr)

        logger.debug("done.")
        return self


    def __getattr__(self, name):
        return CKernel(self.env, self.ffi, getattr(self.program, name))



@decallmethods(timings)
class CKernel(object):

    def __init__(self, env, ffi, kernel):
        self.env = env
        self.ffi = ffi
        self.kernel = kernel
        self.keep_ref = dict()
        self.dev_args = OrderedDict()


    def set_local_memory(self, i, arg):
        pass


    def set_int(self, i, arg):
        self.dev_args[i] = arg


    def set_float(self, i, arg):
        self.dev_args[i] = arg


    def set_array(self, i, arg):
        self.keep_ref[i] = arg
        self.dev_args[i] = self.ffi.cast("REAL *", arg.__array_interface__['data'][0])


    def allocate_buffer(self, i, shape):
        arg = np.zeros(shape, dtype=REAL)
        self.dev_args[i] = self.ffi.cast("REAL *", arg.__array_interface__['data'][0])
        return arg


    def map_buffer(self, i, arr):
        pass


    def run(self):
        args = self.dev_args.values()
        self.kernel(*args)



libkernels = {}
libkernels["c"] = CModule(CEnv(fast_math=True)).build()
libkernels["cl"] = CLModule(CLEnv(fast_math=True)).build(junroll=2)


def get_extension(use_sp=False, use_cl=False):
    libname = "cl" if use_cl else "c"
    prec = "single" if use_sp else "double"
    logger.debug("Using %s precision %s extension module.", prec, libname.upper())
    return libkernels[libname]


use_cl = True if "--use_cl" in sys.argv else False
use_sp = True if "--use_sp" in sys.argv else False


kernels = get_extension(use_sp=use_sp, use_cl=use_cl)


########## end of file ##########
