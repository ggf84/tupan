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


__all__ = ["Extensions"]

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
        src_file = "libcl_gravity.cl"
        fname = os.path.join(path, src_file)
        with open(fname, 'r') as fobj:
            src = fobj.read()
        self.src = src
        self.path = path


    def build(self, junroll=8):
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
#        gsize = 0
#        lsize = 0
#        ngroups = 1
#        lsize_max = self._lsize_max
#        while gsize < ni and lsize < lsize_max:
#            ngroups += 1
##            lsize = ((ni-1)//ngroups + 1)
#            lsize = ((ni-1)//ngroups + 1)
#            gsize = lsize * ngroups

        gsize = ((ni-1)//8 + 1) * 8


#        lsize_max = self._lsize_max
##        ngroups = ((ni-1)//lsize_max + 1)
##        ngroups = ((ni-1) + lsize_max)//lsize_max
##        ngroups = int((ni-1)**0.5 + 1)
##        ngroups = int(ni**0.5)
##        lsize = ((ni-1)//ngroups + 1)
#        lsize = min(lsize_max, (ni+1)//2)
#        ngroups = (ni+1)//lsize
#        gsize = lsize * ngroups

#        self._lsize = lsize
        self._gsize = gsize


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


    def set_output_buffer(self, i, arr):
        memf = cl.mem_flags
        self.dev_buff[i] = cl.Buffer(self.env.ctx,
                                     memf.READ_WRITE | memf.USE_HOST_PTR,
                                     hostbuf=arr)

        arg = self.dev_buff[i]
        self.kernel.set_arg(i, arg)


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

        if prec is "double":
            from tupan.lib import libTupanDP as program
        else:
            from tupan.lib import libTupanSP as program
        self.program = program

        logger.debug("done.")
        return self


    def __getattr__(self, name):
        return CKernel(self.env, getattr(self.program, name))



@decallmethods(timings)
class CKernel(object):

    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel
        self.dev_args = OrderedDict()


    def set_int(self, i, arg):
        self.dev_args[i] = UINT(arg)


    def set_float(self, i, arg):
        self.dev_args[i] = REAL(arg)


    def set_array(self, i, arg):
        self.dev_args[i] = arg


    def set_output_buffer(self, i, arg):
        self.dev_args[i] = arg


    def set_local_memory(self, i, arg):
        pass


    def run(self):
        args = self.dev_args.values()
        self.kernel(*args)



libkernels = {}
libkernels["c"] = CModule(CEnv(fast_math=True)).build()
libkernels["cl"] = CLModule(CLEnv(fast_math=True)).build(junroll=8)


def get_extension(use_sp=False, use_cl=False):
    libname = "cl" if use_cl else "c"
    prec = "single" if use_sp else "double"
    logger.debug("Using %s precision %s extension module.", prec, libname.upper())
    return libkernels[libname]


use_cl = True if "--use_cl" in sys.argv else False
use_sp = True if "--use_sp" in sys.argv else False


kernels = get_extension(use_sp=use_sp, use_cl=use_cl)


########## end of file ##########
