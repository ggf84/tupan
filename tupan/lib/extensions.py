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
logging.basicConfig(filename="spam.log", filemode='w',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        prec = "double" if self.env.dtype.char is 'd' else "single"
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
        self._lsize = None
        self._lsize_max = None
        self._gsize = None
        self.dev_buff = {}
        self.input_buffer = {}
        self.output_buffer = {}
        self.output_shape = {}


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


    def set_local_memory(self, i, arg):
        def foo(x, y): return x * y
        size = self.env.dtype.itemsize * reduce(foo, self.local_size)
        arg = cl.LocalMemory(size * arg)
        self.kernel.set_arg(i, arg)


    def set_int(self, i, arg):
        arg = np.uint32(arg)
        self.kernel.set_arg(i, arg)


    def set_float(self, i, arg):
        arg = self.env.dtype.type(arg)
        self.kernel.set_arg(i, arg)


    def set_input_buffer(self, i, arr):
        arr = self.env.dtype.type(arr)
        def allocate_buffer(arr):
            memf = cl.mem_flags
            mapf = cl.map_flags
            dev_buf = cl.Buffer(self.env.ctx,
                                memf.READ_ONLY,
                                size=arr.nbytes)
            pin_buf = cl.Buffer(self.env.ctx,
                                memf.READ_ONLY | memf.ALLOC_HOST_PTR,
                                size=arr.nbytes)
            (in_buf, ev) = cl.enqueue_map_buffer(self.env.queue, pin_buf,
                                                 mapf.WRITE, 0, arr.shape,
                                                 self.env.dtype, "C")
#            return (dev_buf, in_buf)
            return (pin_buf, in_buf)

        if not i in self.dev_buff:
            (self.dev_buff[i], self.input_buffer[i]) = allocate_buffer(arr)
            logger.debug("%s: allocating buffer for input arg #%d - %s.",
                         self.kernel.function_name, i, self.dev_buff[i])
        if len(arr) > len(self.input_buffer[i]):
            (self.dev_buff[i], self.input_buffer[i]) = allocate_buffer(arr)
            logger.debug("%s: reallocating buffer for input arg #%d - %s.",
                         self.kernel.function_name, i, self.dev_buff[i])

        inbuff = self.input_buffer[i][:len(arr)]
        inbuff[:] = arr
        cl.enqueue_copy(self.env.queue, self.dev_buff[i], inbuff)

#        self.input_buffer[i][:len(arr)] = arr
#        cl.enqueue_copy(self.env.queue, self.dev_buff[i], self.input_buffer[i][:len(arr)])

#        cl.enqueue_copy(self.env.queue, self.dev_buff[i], np.ascontiguousarray(arr, dtype=self.env.dtype))

#        memf = cl.mem_flags
#        self.dev_buff[i] = cl.Buffer(self.env.ctx, memf.READ_ONLY | memf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(arr, dtype=self.env.dtype))
#        self.dev_buff[i] = cl.Buffer(self.env.ctx, memf.READ_ONLY | memf.USE_HOST_PTR, hostbuf=np.ascontiguousarray(arr, dtype=self.env.dtype))

        arg = self.dev_buff[i]
        self.kernel.set_arg(i, arg)


    def set_output_buffer(self, i, arr):
        def allocate_buffer(arr):
            memf = cl.mem_flags
            mapf = cl.map_flags
            dev_buf = cl.Buffer(self.env.ctx,
                                memf.WRITE_ONLY,
                                size=arr.nbytes)
            pin_buf = cl.Buffer(self.env.ctx,
                                memf.WRITE_ONLY | memf.ALLOC_HOST_PTR,
                                size=arr.nbytes)
            (out_buf, ev) = cl.enqueue_map_buffer(self.env.queue, pin_buf,
                                                  mapf.READ, 0, arr.shape,
                                                  self.env.dtype, "C")
#            return (dev_buf, out_buf)
            return (pin_buf, out_buf)

        if not i in self.dev_buff:
            (self.dev_buff[i], self.output_buffer[i]) = allocate_buffer(arr)
            logger.debug("%s: allocating buffer for output arg #%d - %s.",
                         self.kernel.function_name, i, self.dev_buff[i])
        if arr.shape > self.output_buffer[i].shape:
            (self.dev_buff[i], self.output_buffer[i]) = allocate_buffer(arr)
            logger.debug("%s: reallocating buffer for output arg #%d - %s.",
                         self.kernel.function_name, i, self.dev_buff[i])

        self.output_shape[i] = arr.shape
        arg = self.dev_buff[i]
        self.kernel.set_arg(i, arg)


    def run(self):
        cl.enqueue_nd_range_kernel(self.env.queue,
                                   self.kernel,
                                   self.global_size,
                                   self.local_size,
                                  ).wait()


    def get_result(self):
#        def getter(item):
#            (i, shape) = item
#            return self.dev_buff[i].get_host_array(shape, self.env.dtype)

        def getter(item):
            (i, shape) = item
            cl.enqueue_copy(self.env.queue, self.output_buffer[i], self.dev_buff[i])
            return self.output_buffer[i][:shape[0]]

        return map(getter, self.output_shape.items())



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
        prec = "double" if self.env.dtype.char is 'd' else "single"
        logger.debug("Building %s precision C extension module.", prec)

        if prec is "double":
            from tupan.lib import libc64_gravity as program
        else:
            from tupan.lib import libc32_gravity as program
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
        self.dev_args = {}
        self.res_ids = {}


    def set_int(self, i, arg):
        self.dev_args[i] = np.uint32(arg)


    def set_float(self, i, arg):
        self.dev_args[i] = self.env.dtype.type(arg)


    def set_input_buffer(self, i, arg):
        if not hasattr(arg, "astype"):
            self.dev_args[i] = self.env.dtype.type(arg)
        else:
            self.dev_args[i] = arg.astype(self.env.dtype.type)


    def set_output_buffer(self, i, arg):
        if not hasattr(arg, "astype"):
            self.dev_args[i] = self.env.dtype.type(arg)
        else:
            self.dev_args[i] = arg.astype(self.env.dtype.type)
        self.res_ids[i] = i


    def set_local_memory(self, i, arg):
        pass


    def run(self):
        args = self.dev_args.values()
        self.dev_result = self.kernel(*args)    # self.dev_result == None


    def get_result(self):       # XXX: FIXME
        return [self.dev_args[i] for i in self.res_ids.values()]




libkernels = {"sp": {"c": None, "cl": None}, "dp": {"c": None, "cl": None}}
libkernels["sp"]["c"] = CModule(CEnv(dtype='f', fast_math=True)).build()
libkernels["sp"]["cl"] = CLModule(CLEnv(dtype='f', fast_math=True)).build(junroll=8)
libkernels["dp"]["c"] = CModule(CEnv(dtype='d', fast_math=True)).build()
libkernels["dp"]["cl"] = CLModule(CLEnv(dtype='d', fast_math=True)).build(junroll=8)


def get_extension(use_sp=False, use_cl=False):
    if use_sp:
        prec = "single"
        if use_cl:
            logger.debug("Using %s precision CL extension module.", prec)
            return libkernels["sp"]["cl"]
        else:
            logger.debug("Using %s precision C extension module.", prec)
            return libkernels["sp"]["c"]
    else:
        prec = "double"
        if use_cl:
            logger.debug("Using %s precision CL extension module.", prec)
            return libkernels["dp"]["cl"]
        else:
            logger.debug("Using %s precision C extension module.", prec)
            return libkernels["dp"]["c"]


use_cl = True if "--use_cl" in sys.argv else False
use_sp = True if "--use_sp" in sys.argv else False

kernels = get_extension(use_sp=use_sp, use_cl=use_cl)


########## end of file ##########
