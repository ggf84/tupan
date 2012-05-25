#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import os
import sys
import logging
from warnings import warn
from functools import reduce
import numpy as np
import pyopencl as cl
from .utils.timing import decallmethods, timings


__all__ = ["Extensions"]

logger = logging.getLogger(__name__)
logging.basicConfig(filename='spam.log', filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


class CLEnv(object):

    def __init__(self, dtype='d', fast_math=True):
        self.dtype = np.dtype(dtype)
        self.fast_math = fast_math
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)



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



class CLKernel(object):

    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel


    def set_args(self, *args, **kwargs):
        # Get keyword arguments
        lsize = kwargs.pop("local_size")
        local_size = lsize if isinstance(lsize, tuple) else (lsize,)

        gsize = kwargs.pop("global_size")
        global_size = gsize if isinstance(gsize, tuple) else (gsize,)

        result_shape = kwargs.pop("result_shape")
        local_memory_shape = kwargs.pop("local_memory_shape")

        if kwargs:
            msg = "{0}.load_data received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        # Set input buffers and kernel args on CL device
        mf = cl.mem_flags
        dev_args = [global_size, local_size]
        for arg in args:
            if isinstance(arg, np.ndarray):
                hostbuf = np.ascontiguousarray(arg, dtype=self.env.dtype)
                item = cl.Buffer(self.env.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hostbuf)
            elif isinstance(arg, float):
                item = self.env.dtype.type(arg)
            elif isinstance(arg, int):
                item = np.uint32(arg)
            else:
                raise TypeError()
            dev_args.append(item)

        # Set output buffer on CL device
        self.host_result = np.empty(result_shape, dtype=self.env.dtype)
        self.device_result = cl.Buffer(self.env.ctx, mf.WRITE_ONLY, size=self.host_result.nbytes)
        dev_args.append(self.device_result)

        # Set local memory sizes on CL device
        def foo(x, y): return x * y
        base_mem_size = reduce(foo, local_size)
        base_mem_size *= self.env.dtype.itemsize
        local_mem_size_list = (i * base_mem_size for i in local_memory_shape)
        for size in local_mem_size_list:
            dev_args.append(cl.LocalMemory(size))

        # Finally puts everything in _kernelargs
        self.dev_args = dev_args


    def run(self):
        args = self.dev_args
        self.kernel(self.env.queue, *args).wait()


    def get_result(self):
        cl.enqueue_copy(self.env.queue, self.host_result, self.device_result)
        return self.host_result




class CEnv(object):

    def __init__(self, dtype='d', fast_math=True):
        self.dtype = np.dtype(dtype)
        self.fast_math = fast_math



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



class CKernel(object):

    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel


    def set_args(self, *args, **kwargs):
        dev_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                hostbuf = np.ascontiguousarray(arg, dtype=self.env.dtype)
                item = hostbuf
            elif isinstance(arg, float):
                item = self.env.dtype.type(arg)
            elif isinstance(arg, int):
                item = np.uint32(arg)
            else:
                raise TypeError()
            dev_args.append(item)
        self.dev_args = dev_args


    def run(self):
        args = self.dev_args
        self.device_result = self.kernel(*args)


    def get_result(self):
        self.host_result = self.device_result
        return self.host_result




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
