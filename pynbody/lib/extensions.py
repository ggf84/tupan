#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import os
import sys
import logging
from functools import reduce
import numpy as np
import pyopencl as cl
from .utils.timing import decallmethods, timings


__all__ = ["Extensions"]

logger = logging.getLogger(__name__)
logging.basicConfig(filename='spam.log', filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


@decallmethods(timings)
class CLEnv(object):

    def __init__(self, dtype='d', fast_math=True):
        self.dtype = np.dtype(dtype)
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
        prec = 'double' if self.env.dtype.char is 'd' else 'single'
        logger.debug("Building %s precision CL extension module.", prec)

        # setting options
        options = " -I {path}".format(path=self.path)
        options += " -D JUNROLL={junroll}".format(junroll=junroll)
        if self.env.dtype.char is 'd':
            options += " -D DOUBLE"
        if self.env.fast_math:
            options += " -cl-fast-relaxed-math"

        # building program
        self.program = cl.Program(self.env.ctx, self.src).build(options=options)

        return self


    def __getattr__(self, name):
        return CLKernel(self.env, getattr(self.program, name))



@decallmethods(timings)
class CLKernel(object):

    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel
        self._lsize = None
        self._lsize_max = None
        self._gsize = None
        self.dev_args = {}


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
        lsize = self._lsize_max
        lsize = min(lsize, ((ni-1)//8 + 1))
        self._lsize = lsize
        self._gsize = ((ni-1)//lsize + 1) * lsize


    def set_arg(self, mode, i, arg):
        """
        mode: 'IN', 'OUT', 'LMEM'
        i: arg index
        arg: arg value
        """
        if mode is 'IN':
            if isinstance(arg, int):
                self.dev_args[i] = np.uint32(arg)
            elif isinstance(arg, float):
                self.dev_args[i] = self.env.dtype.type(arg)
            elif isinstance(arg, np.ndarray):
                hostbuf = np.ascontiguousarray(arg, dtype=self.env.dtype)
#                if not i in self.dev_args:
#                    mf = cl.mem_flags
#                    self.dev_args[i] = cl.Buffer(self.env.ctx, mf.READ_ONLY, size=hostbuf.nbytes)
#                    print('IN:', self.kernel.function_name, self.dev_args)
#                if hostbuf.nbytes > self.dev_args[i].size:
#                    mf = cl.mem_flags
#                    self.dev_args[i] = cl.Buffer(self.env.ctx, mf.READ_ONLY, size=hostbuf.nbytes)
#                    print('IN realocation:', self.kernel.function_name, self.dev_args)
#                cl.enqueue_copy(self.env.queue, self.dev_args[i], hostbuf)
                mf = cl.mem_flags
#                self.dev_args[i] = cl.Buffer(self.env.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hostbuf)
                self.dev_args[i] = cl.Buffer(self.env.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=hostbuf)
            else:
                raise TypeError()
        elif mode is 'OUT':
            if not i in self.dev_args:
                def foo(x, y): return x * y
                size = self.env.dtype.itemsize * reduce(foo, arg)
                self._shape = arg
                mf = cl.mem_flags
                self.dev_args[i] = cl.Buffer(self.env.ctx, mf.WRITE_ONLY, size=size)
                self.dev_result = self.dev_args[i]
                print('OUT:', self.kernel.function_name, self.dev_args)
            if arg > self._shape:
                def foo(x, y): return x * y
                size = self.env.dtype.itemsize * reduce(foo, arg)
                self._shape = arg
                mf = cl.mem_flags
                self.dev_args[i] = cl.Buffer(self.env.ctx, mf.WRITE_ONLY, size=size)
                self.dev_result = self.dev_args[i]
                print('OUT realocation:', self.kernel.function_name, self.dev_args)
            self.host_shape = arg
        elif mode is 'LMEM':
            def foo(x, y): return x * y
            base = self.env.dtype.itemsize * reduce(foo, self.local_size)
            self.dev_args[i] = cl.LocalMemory(base * arg)


    def run(self):
        for i, arg in self.dev_args.items():
            self.kernel.set_arg(i, arg)
        cl.enqueue_nd_range_kernel(self.env.queue,
                                   self.kernel,
                                   self.global_size,
                                   self.local_size,
                                  ).wait()


    def get_result(self):
        return self.dev_result.get_host_array(self.host_shape, self.env.dtype)




@decallmethods(timings)
class CEnv(object):

    def __init__(self, dtype='d', fast_math=True):
        self.dtype = np.dtype(dtype)
        self.fast_math = fast_math



@decallmethods(timings)
class CModule(object):

    def __init__(self, env):
        self.env = env


    def build(self):
        prec = 'double' if self.env.dtype.char is 'd' else 'single'
        logger.debug("Building %s precision C extension module.", prec)

        if self.env.dtype.char is 'd':
            from pynbody.lib import libc64_gravity as program
        else:
            from pynbody.lib import libc32_gravity as program
        self.program = program

        return self


    def __getattr__(self, name):
        return CKernel(self.env, getattr(self.program, name))



@decallmethods(timings)
class CKernel(object):

    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel
        self.dev_args = {}


    def set_arg(self, mode, i, arg):
        """
        mode: 'IN', 'OUT', 'LMEM'
        i: arg index
        arg: arg value
        """
        if mode is 'IN':
            if isinstance(arg, int):
                self.dev_args[i] = np.uint32(arg)
            elif isinstance(arg, float):
                self.dev_args[i] = self.env.dtype.type(arg)
            elif isinstance(arg, np.ndarray):
                hostbuf = np.ascontiguousarray(arg, dtype=self.env.dtype)
                self.dev_args[i] = hostbuf
            else:
                raise TypeError()
        elif mode is 'OUT':
            pass
        elif mode is 'LMEM':
            pass


    def run(self):
        args = self.dev_args.values()
        self.dev_result = self.kernel(*args)


    def get_result(self):
        return self.dev_result




libkernels = {'sp': {'c': None, 'cl': None}, 'dp': {'c': None, 'cl': None}}
libkernels['sp']['c'] = CModule(CEnv(dtype='f', fast_math=True)).build()
libkernels['sp']['cl'] = CLModule(CLEnv(dtype='f', fast_math=True)).build(junroll=8)
libkernels['dp']['c'] = CModule(CEnv(dtype='d', fast_math=True)).build()
libkernels['dp']['cl'] = CLModule(CLEnv(dtype='d', fast_math=True)).build(junroll=8)


def get_extension(use_sp=False, use_cl=False):
    if use_sp:
        prec = 'single'
        if use_cl:
            logger.debug("Using %s precision CL extension module.", prec)
            return libkernels['sp']['cl']
        else:
            logger.debug("Using %s precision C extension module.", prec)
            return libkernels['sp']['c']
    else:
        prec = 'double'
        if use_cl:
            logger.debug("Using %s precision CL extension module.", prec)
            return libkernels['dp']['cl']
        else:
            logger.debug("Using %s precision C extension module.", prec)
            return libkernels['dp']['c']


use_cl = True if '--use_cl' in sys.argv else False
use_sp = True if '--use_sp' in sys.argv else False

ext = get_extension(use_sp=use_sp, use_cl=use_cl)


########## end of file ##########
