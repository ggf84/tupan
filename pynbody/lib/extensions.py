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
        self._in = {}
        self._out = {}


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


    def _set_arg_in_int(self, i, arg):
        self.dev_args[i] = np.uint32(arg)

    def _set_arg_in_float(self, i, arg):
        self.dev_args[i] = self.env.dtype.type(arg)


    def _allocate_in_buffers(self, arg):
        memf = cl.mem_flags
        mapf = cl.map_flags
        dev_buf = cl.Buffer(self.env.ctx,
                            memf.READ_ONLY,
                            size=arg.nbytes)
        pin_buff = cl.Buffer(self.env.ctx,
                             memf.READ_ONLY | memf.ALLOC_HOST_PTR,
                             size=arg.nbytes)
        (in_data, ev) = cl.enqueue_map_buffer(self.env.queue, pin_buff,
                                              mapf.WRITE, 0, arg.shape,
                                              self.env.dtype, 'C')
        in_shape = in_data.shape
        return (dev_buf, {"data": in_data, "shape": in_shape})

    def _set_arg_in_ndarray(self, i, arg):
        if not i in self.dev_args:
            (self.dev_args[i], self._in[i]) = self._allocate_in_buffers(arg)
            print('IN:', self.kernel.function_name, self.dev_args)
        if len(arg) > len(self._in[i]["data"]):
            (self.dev_args[i], self._in[i]) = self._allocate_in_buffers(arg)
            print('IN realocation:', self.kernel.function_name, self.dev_args)

        self._in[i]["shape"] = arg.shape
        self._in[i]["data"][:len(arg)] = arg
        cl.enqueue_copy(self.env.queue, self.dev_args[i], self._in[i]["data"][:len(arg)])

#        cl.enqueue_copy(self.env.queue, self.dev_args[i], np.ascontiguousarray(arg, dtype=self.env.dtype))

#        memf = cl.mem_flags
#        self.dev_args[i] = cl.Buffer(self.env.ctx, memf.READ_ONLY | memf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(arg, dtype=self.env.dtype))
#        self.dev_args[i] = cl.Buffer(self.env.ctx, memf.READ_ONLY | memf.USE_HOST_PTR, hostbuf=np.ascontiguousarray(arg, dtype=self.env.dtype))


    def _allocate_out_buffers(self, arg):
        memf = cl.mem_flags
        mapf = cl.map_flags
        dev_buf = cl.Buffer(self.env.ctx,
                            memf.WRITE_ONLY,
                            size=arg.nbytes)
        pin_buff = cl.Buffer(self.env.ctx,
                             memf.WRITE_ONLY | memf.ALLOC_HOST_PTR,
                             size=arg.nbytes)
        (out_data, ev) = cl.enqueue_map_buffer(self.env.queue, pin_buff,
                                               mapf.READ, 0, arg.shape,
                                               self.env.dtype, 'C')
        out_shape = out_data.shape
        return (dev_buf, {"data": out_data, "shape": out_shape})

    def _set_arg_out_ndarray(self, i, arg):
        arg = np.empty(arg, dtype=self.env.dtype)
        if not i in self.dev_args:
            (self.dev_args[i], self._out[i]) = self._allocate_out_buffers(arg)
            print('OUT:', self.kernel.function_name, self.dev_args)
        if len(arg) > len(self._out[i]["data"]):
            (self.dev_args[i], self._out[i]) = self._allocate_out_buffers(arg)
            print('OUT realocation:', self.kernel.function_name, self.dev_args)
        self._out[i]["shape"] = arg.shape


    def _set_arg_in_lmem(self, i, arg):
            def foo(x, y): return x * y
            size = self.env.dtype.itemsize * reduce(foo, self.local_size)
            self.dev_args[i] = cl.LocalMemory(size * arg)


    def set_arg(self, mode, i, arg):
        """
        mode: 'IN', 'OUT', 'LMEM'
        i: arg index
        arg: arg value
        """
        if mode is 'IN':
            if isinstance(arg, int):
                self._set_arg_in_int(i, arg)
            elif isinstance(arg, float):
                self._set_arg_in_float(i, arg)
            elif isinstance(arg, np.ndarray):
                self._set_arg_in_ndarray(i, arg)
            else:
                raise TypeError("CLKernel.set_arg recived unexpected argument type: {}".format(type(arg)))
        elif mode is 'OUT':
            if isinstance(arg, int):
                raise NotImplementedError()
            elif isinstance(arg, float):
                raise NotImplementedError()
            elif isinstance(arg, tuple):
                self._set_arg_out_ndarray(i, arg)
            else:
                raise TypeError("CLKernel.set_arg recived unexpected argument type: {}".format(type(arg)))
        elif mode is 'LMEM':
            self._set_arg_in_lmem(i, arg)
        else:
            raise TypeError("CLKernel.set_arg recived unexpected mode setting: {}".format(mode))


    def run(self):
        for i, arg in self.dev_args.items():
            self.kernel.set_arg(i, arg)
        cl.enqueue_nd_range_kernel(self.env.queue,
                                   self.kernel,
                                   self.global_size,
                                   self.local_size,
                                  ).wait()


    def get_result(self):
#        return self.dev_result.get_host_array(self.host_shape, self.env.dtype)
        ret = []
        for i, _out in self._out.items():
            ret.append(self.dev_args[i].get_host_array(_out["shape"], self.env.dtype))
#            cl.enqueue_copy(self.env.queue, _out["data"], self.dev_args[i])
#            ret.append(_out["data"][:_out["shape"][0]])
        return ret[0]


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
