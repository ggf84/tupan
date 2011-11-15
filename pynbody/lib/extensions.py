#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import os
import sys
from warnings import warn
from functools import reduce
import numpy as np
import pyopencl as cl


__all__ = ["Extensions"]


def get_pattern(pattern, src_code):
    for line in src_code.splitlines():
        if pattern in line:
            return line.split()[-1]


class Extensions(object):
    """

    """
    def __init__(self, path):
        self.path = path
        self.ext_type = None
        self._cl_ctx = None
        self._cl_queue = None
        self._cl_devbuf_res = None


    # --------------------------------------------------------------------------
    # Load source

    def load_source(self, filename):
        """
        Read source files as a string
        """
        self.src_name = os.path.splitext(filename)

        # read in source code and output shape
        fname = os.path.join(self.path, filename)
        with open(fname, 'r') as fobj:
            self.src_code = fobj.read()

        # read flops count from core function
        fname = os.path.join(self.path, self.src_name[0]+"_core.h")
        with open(fname, 'r') as fobj:
            self.flops_count = int(get_pattern("Total flop count", fobj.read()))



    # --------------------------------------------------------------------------
    # Build program

    def _c_build(self, **kwargs):
        ctype = kwargs.pop("dtype", 'f')
        if not (ctype is 'd' or ctype is 'f'):
            msg = "{0}._cl_build received unexpected dtype: '{1}'."
            raise TypeError(msg.format(self.__class__.__name__, ctype))

        dummy = kwargs.pop("junroll", 1)    # Only meaningful on CL kernels.

        if kwargs:
            msg = "{0}._cl_build received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        cppargs = []
        signature="lib32_"+self.src_name[0]
        self.dtype = np.dtype(ctype)
        if self.dtype is np.dtype('d'):
            cppargs.append("-D DOUBLE")
            signature = signature.replace("lib32_", "lib64_")

        try:
            from instant import build_module
            prog = build_module(source_directory=self.path,
                                code=self.src_code,
                                init_code='import_array();',
                                system_headers=["numpy/arrayobject.h"],
                                include_dirs=[np.get_include(),self.path],
                                libraries=["m"],
                                cppargs=cppargs,
                                signature=signature)
            self.program = getattr(prog, self.src_name[0])
            self.ext_type = "C_EXTENSION"
        except Exception as c_build_except:
            warn("{0}".format(c_build_except), stacklevel=2)
            msg = "Sorry, I can't build C extension '{0}'.\nExiting..."
            print(msg.format(''.join(self.src_name)))
            sys.exit(0)


    def _cl_build(self, **kwargs):
        ctype = kwargs.pop("dtype", 'f')
        if not (ctype is 'd' or ctype is 'f'):
            msg = "{0}._cl_build received unexpected dtype: '{1}'."
            raise TypeError(msg.format(self.__class__.__name__, ctype))

        junroll = kwargs.pop("junroll", 1)

        if kwargs:
            msg = "{0}._cl_build received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        options = " -I {path}".format(path=self.path)
        options += " -D JUNROLL={junroll}".format(junroll=junroll)

        self.dtype = np.dtype(ctype)
        if self.dtype is np.dtype('d'):
            options += " -D DOUBLE"

        options += " -cl-fast-relaxed-math"

        try:
            self._cl_ctx = cl.create_some_context()
            self._cl_queue = cl.CommandQueue(self._cl_ctx)
            prog = cl.Program(self._cl_ctx, self.src_code).build(options=options)
            self.program = getattr(prog, self.src_name[0])
            self.ext_type = "CL_EXTENSION"
        except Exception as cl_build_except:
            warn("{0}".format(cl_build_except), RuntimeWarning, stacklevel=2)
            msg = "Sorry, I can't build OpenCL extension '{0}'."
            print(msg.format(''.join(self.src_name)))
            raise RuntimeWarning


    def build(self, **kwargs):
        if ".c" in self.src_name:
            self._c_build(**kwargs)
        elif ".cl" in self.src_name:
            try:
                self._cl_build(**kwargs)
            except RuntimeWarning:
                ans = raw_input("Do you want to continue with C extension ([y]/n)? ")
                ans = ans.lower()
                if ans == 'n' or ans == 'no':
                    print('Exiting...')
                    sys.exit(0)
                self.load_source(self.src_name[0]+".c")
                self._c_build(**kwargs)



    # --------------------------------------------------------------------------
    # Load data

    def _c_load_data(self, *args):
        self._kernelargs = (args,)


    def _cl_load_data(self, *args, **kwargs):
        # Get keyword arguments
        lsize = kwargs.pop("local_size")
        local_size = lsize if isinstance(lsize, tuple) else (lsize,)

        gsize = kwargs.pop("global_size")
        global_size = gsize if isinstance(gsize, tuple) else (gsize,)

        output_buf = kwargs.pop("output_buf")

        if kwargs:
            msg = "{0}._cl_load_data received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        # Set input buffers and kernel args on CL device
        mf = cl.mem_flags
        dev_args = [global_size, local_size]
        for item in args:
            if isinstance(item, np.ndarray):
                hostbuf = item.copy().astype(self.dtype)
                dev_args.append(cl.Buffer(self._cl_ctx,
                                          mf.READ_ONLY | mf.COPY_HOST_PTR,
                                          hostbuf=hostbuf))
            else:
                dev_args.append(item)

        # Set output buffer on CL device
        self._hostbuf_output = output_buf.copy().astype(self.dtype)
        self._cl_devbuf_res = cl.Buffer(self._cl_ctx,
                                        mf.WRITE_ONLY | mf.USE_HOST_PTR,
                                        hostbuf=self._hostbuf_output)
        dev_args.append(self._cl_devbuf_res)

        # Set local memory sizes on CL device
        base_mem_size = reduce(lambda x, y: x * y, local_size)
        base_mem_size *= self.dtype.itemsize
        local_mem_size_list = (4*base_mem_size, base_mem_size)
        for size in local_mem_size_list:
            dev_args.append(cl.LocalMemory(size))

        # Finally puts everything in _kernelargs
        self._kernelargs = dev_args


    def load_data(self, *args, **kwargs):
        if self.ext_type is "C_EXTENSION":
            self._c_load_data(*args)
        elif self.ext_type is "CL_EXTENSION":
            self._cl_load_data(*args, **kwargs)



    # --------------------------------------------------------------------------
    # Execute program

    def _c_execute(self):
        args = self._kernelargs
        self._hostbuf_output = self.program(*args)


    def _cl_execute(self):
        args = self._kernelargs
        self.program(self._cl_queue, *args).wait()


    def execute(self):
        if self.ext_type is "C_EXTENSION":
            self._c_execute()
        elif self.ext_type is "CL_EXTENSION":
            self._cl_execute()



    # --------------------------------------------------------------------------
    # Get result

    def _c_get_result(self):
        return self._hostbuf_output


    def _cl_get_result(self):
        cl.enqueue_copy(self._cl_queue, self._hostbuf_output, self._cl_devbuf_res)
        return self._hostbuf_output


    def get_result(self):
        if self.ext_type is "C_EXTENSION":
            return self._c_get_result()
        elif self.ext_type is "CL_EXTENSION":
            return self._cl_get_result()





def build_kernels():
    dirname = os.path.dirname(__file__)
    abspath = os.path.abspath(dirname)
    path = os.path.join(abspath, "ext2")

    sources = ["p2p_phi_kernel.cl", "p2p_acc_kernel.cl",
               "p2p_phi_kernel.c", "p2p_acc_kernel.c",
               "p2p_pnacc_kernel.c"]

    kernels = {}

    # build double precision kernels
    print("#"*40, file=sys.stderr)
    print("### building double precision kernels...", file=sys.stderr)
    for src in sources:
        name, ext = os.path.splitext(src)
        ext = ext.strip('.')
        key = ext+"_lib64_"+name
        kernels[key] = Extensions(path)
        kernels[key].load_source(src)
        kernels[key].build(dtype='d', junroll=16)
    print("### ...done.", file=sys.stderr)
    print("#"*40, file=sys.stderr)

    # build single precision kernels
    print("#"*40, file=sys.stderr)
    print("### building single precision kernels...", file=sys.stderr)
    for src in sources:
        name, ext = os.path.splitext(src)
        ext = ext.strip('.')
        key = ext+"_lib32_"+name
        kernels[key] = Extensions(path)
        kernels[key].load_source(src)
        kernels[key].build(dtype='f', junroll=16)
    print("### ...done.", file=sys.stderr)
    print("#"*40, file=sys.stderr)

    return kernels


########## end of file ##########
