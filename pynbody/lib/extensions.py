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


@decallmethods(timings)
class CLExtensions(object):
    """

    """
    def __init__(self, *args, **kwargs):
        ctype = kwargs.pop("dtype", 'd')
        self._dtype = np.dtype(ctype)
        self._junroll = kwargs.pop("junroll", 8)

        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        dirname = os.path.dirname(__file__)
        abspath = os.path.abspath(dirname)
        self._path = os.path.join(abspath, "src")
        self._cl_source_name = "libcl_gravity.cl"


    def build_kernels(self):
        import pyopencl as cl

        # read in source code and output shape
        fname = os.path.join(self._path, self._cl_source_name)
        with open(fname, 'r') as fobj:
            source_code = fobj.read()

        # setting options
        options = " -I {path}".format(path=self._path)
        options += " -D JUNROLL={junroll}".format(junroll=self._junroll)
        if self._dtype is np.dtype('d'):
            options += " -D DOUBLE"
        options += " -cl-fast-relaxed-math"

        # build CL program
        self._cl_ctx = cl.create_some_context()
        self._cl_queue = cl.CommandQueue(self._cl_ctx)
        program = cl.Program(self._cl_ctx, source_code).build(options=options)
        self.libcl = program


    def get_kernel(self, kernel_name):
        self.kernel_func = getattr(self.libcl, kernel_name)
        return self


    def set_kernel_args(self, *args, **kwargs):
        # Get keyword arguments
        lsize = kwargs.pop("local_size")
        local_size = lsize if isinstance(lsize, tuple) else (lsize,)

        gsize = kwargs.pop("global_size")
        global_size = gsize if isinstance(gsize, tuple) else (gsize,)

        output_layout = kwargs.pop("output_layout")
        lmem_layout = kwargs.pop("lmem_layout")

        if kwargs:
            msg = "{0}.load_data received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        # Set input buffers and kernel args on CL device
        mf = cl.mem_flags
        dev_args = [global_size, local_size]
        for item in args:
            if isinstance(item, np.ndarray):
                hostbuf = item.copy().astype(self._dtype)
                dev_args.append(cl.Buffer(self._cl_ctx,
                                          mf.READ_ONLY | mf.COPY_HOST_PTR,
                                          hostbuf=hostbuf))
            elif isinstance(item, np.floating):
                dev_args.append(item.astype(self._dtype))
            else:
                dev_args.append(item)

        # Set output buffer on CL device
        self.kernel_result = np.empty(output_layout, dtype=self._dtype)
        self._cl_devbuf_res = cl.Buffer(self._cl_ctx,
                                        mf.WRITE_ONLY | mf.USE_HOST_PTR,
                                        hostbuf=self.kernel_result)
        dev_args.append(self._cl_devbuf_res)

        # Set local memory sizes on CL device
        def foo(x, y): return x * y
        base_mem_size = reduce(foo, local_size)
        base_mem_size *= self._dtype.itemsize
        local_mem_size_list = (i * base_mem_size for i in lmem_layout)
        for size in local_mem_size_list:
            dev_args.append(cl.LocalMemory(size))

        # Finally puts everything in _kernelargs
        self.kernel_args = dev_args


    def run(self):
        args = self.kernel_args
        self.kernel_func(self._cl_queue, *args).wait()


    def get_result(self):
        cl.enqueue_copy(self._cl_queue, self.kernel_result, self._cl_devbuf_res)
        return self.kernel_result



@decallmethods(timings)
class CExtensions(object):
    """

    """
    def __init__(self, *args, **kwargs):
        ctype = kwargs.pop("dtype", 'd')
        self._dtype = np.dtype(ctype)

        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))


    def build_kernels(self):
        if self._dtype is np.dtype('d'):
            from pynbody.lib import libc64_gravity as libc_gravity
        else:
            from pynbody.lib import libc32_gravity as libc_gravity
        self.libc = libc_gravity


    def get_kernel(self, kernel_name):
        self.kernel_func = getattr(self.libc, kernel_name)
        return self


    def set_kernel_args(self, *args, **kwargs):
        self.kernel_args = args


    def run(self):
        args = self.kernel_args
        self.kernel_result = self.kernel_func(*args)


    def get_result(self):
        return self.kernel_result



@decallmethods(timings)
class Extensions(object):
    """

    """
    def __init__(self, *args, **kwargs):
        self.dtype = kwargs.pop("dtype", 'd')
        self.junroll = kwargs.pop("junroll", 8)

        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        self.cext = CExtensions(dtype=self.dtype)
        self.clext = CLExtensions(dtype=self.dtype, junroll=self.junroll)


    def build_kernels(self, use_cl=False):
        logger.debug("Building kernels...")
        self.cext.build_kernels()
        try:
            self.clext.build_kernels()
            has_cl = True
        except Exception as exc:
            has_cl = False
            logger.exception(exc)

        if not use_cl:
            self.extension = self.cext
            logger.debug("Using C extensions.")
        elif use_cl and has_cl:
            self.extension = self.clext
            logger.debug("Using CL extensions.")
        if use_cl and not has_cl:
            msg = ("Sorry, a problem occurred while trying to build OpenCL extensions."
                   "\nDo you want to continue with C extensions ([y]/n)? ")
            ans = raw_input(msg)
            ans = ans.lower()
            if ans == 'n' or ans == 'no':
                print('Exiting...', file=sys.stderr)
                sys.exit(0)
            self.extension = self.cext
            logger.debug("Using C extensions.")

    def get_kernel(self, kernel_name):
        return self.extension.get_kernel(kernel_name)


kernel_library = Extensions(dtype='d', junroll=8)


########## end of file ##########
