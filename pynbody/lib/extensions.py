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
        self._path = os.path.join(abspath, "ext")
        self._cl_source_name = "gravity_kernels.cl"


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

        output_buf = kwargs.pop("output_buf")
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
        self.kernel_result = output_buf.copy().astype(self._dtype)
        self._cl_devbuf_res = cl.Buffer(self._cl_ctx,
                                        mf.WRITE_ONLY | mf.USE_HOST_PTR,
                                        hostbuf=self.kernel_result)
        dev_args.append(self._cl_devbuf_res)

        # Set local memory sizes on CL device
        base_mem_size = reduce(lambda x, y: x * y, local_size)
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




class NewExtensions(object):
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


    def build_kernels(self, device='cpu'):
        print("building kernels...")
        self.cext.build_kernels()
        try:
            self.clext.build_kernels()
            has_cl = True
        except:
            has_cl = False

        if device == 'cpu':
            self.extension = self.cext
            print("Using C extensions", file=sys.stderr)
        elif device == 'gpu' and has_cl:
            self.extension = self.clext
            print("Using CL extensions", file=sys.stderr)
        if device == 'gpu' and not has_cl:
            msg = ("Sorry, a problem occurred while trying to build OpenCL extensions."
                   "\nDo you want to continue with C extensions ([y]/n)? ")
            ans = raw_input(msg)
            ans = ans.lower()
            if ans == 'n' or ans == 'no':
                print('Exiting...', file=sys.stderr)
                sys.exit(0)
            self.extension = self.cext
            print("Using C extensions", file=sys.stderr)


    def get_kernel(self, kernel_name):
        return self.extension.get_kernel(kernel_name)

















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



    # --------------------------------------------------------------------------
    # Build program

    def _c_build(self, **kwargs):
        ctype = kwargs.pop("dtype", 'f')
        if not (ctype is 'd' or ctype is 'f'):
            msg = "{0}._c_build received unexpected dtype: '{1}'."
            raise TypeError(msg.format(self.__class__.__name__, ctype))

        dummy = kwargs.pop("junroll", 1)    # Only meaningful on CL kernels.

        if kwargs:
            msg = "{0}._c_build received unexpected keyword arguments: {1}."
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
            self.program = build_module(source_directory=self.path,
                                        code=self.src_code,
                                        init_code='import_array();',
                                        system_headers=["numpy/arrayobject.h"],
                                        include_dirs=[np.get_include(),self.path],
                                        libraries=["m"],
                                        cppargs=cppargs,
                                        signature=signature)
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
            self.program = cl.Program(self._cl_ctx, self.src_code).build(options=options, cache_dir="/tmp/myopencl_cache")
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



    def get_kernel(self, kernel_name):
        import copy
        self.kernel_func = getattr(self.program, kernel_name)
        self.kernel_name = kernel_name
        # read flops count from core function
        fname = os.path.join(self.path, kernel_name+"_core.h")
        with open(fname, 'r') as fobj:
            self.flops_count = int(get_pattern("Total flop count", fobj.read()))
        return copy.copy(self)



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

        lmem_layout = kwargs.pop("lmem_layout")

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
            elif isinstance(item, np.floating):
                dev_args.append(item.astype(self.dtype))
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
        local_mem_size_list = (i * base_mem_size for i in lmem_layout)
        for size in local_mem_size_list:
            dev_args.append(cl.LocalMemory(size))

        # Finally puts everything in _kernelargs
        self._kernelargs = dev_args


    def set_kernel_args(self, *args, **kwargs):
        if self.ext_type is "C_EXTENSION":
            self._c_load_data(*args)
        elif self.ext_type is "CL_EXTENSION":
            self._cl_load_data(*args, **kwargs)



    # --------------------------------------------------------------------------
    # Execute program

    def _c_execute(self):
        args = self._kernelargs
        self._hostbuf_output = self.kernel_func(*args)


    def _cl_execute(self):
        args = self._kernelargs
        self.kernel_func(self._cl_queue, *args).wait()


    def run(self):
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





def build_kernels(device='gpu'):
    dirname = os.path.dirname(__file__)
    abspath = os.path.abspath(dirname)
    path = os.path.join(abspath, "ext")

    kernels = {}

    print("building C kernels...", file=sys.stderr)
    extension_program = Extensions(path)
    extension_program.load_source("gravity_kernels.c")
    extension_program.build(dtype='d', junroll=8)
    for name in ("p2p_phi_kernel", "p2p_acc_kernel", "p2p_pnacc_kernel"):
        kernels["c_lib64_"+name] = extension_program.get_kernel(name)
    extension_program.build(dtype='f', junroll=8)
    for name in ("p2p_phi_kernel", "p2p_acc_kernel", "p2p_pnacc_kernel"):
        kernels["c_lib32_"+name] = extension_program.get_kernel(name)

    print("building CL kernels...", file=sys.stderr)
    extension_program = Extensions(path)
    extension_program.load_source("gravity_kernels.cl")
    extension_program.build(dtype='d', junroll=8)
    for name in ("p2p_phi_kernel", "p2p_acc_kernel"):
        kernels["cl_lib64_"+name] = extension_program.get_kernel(name)
    extension_program.build(dtype='f', junroll=8)
    for name in ("p2p_phi_kernel", "p2p_acc_kernel"):
        kernels["cl_lib32_"+name] = extension_program.get_kernel(name)


    all_kernels = kernels
    kernels = {}
    if device == 'gpu':
        kernels['p2p_phi_kernel'] = all_kernels['cl_lib64_p2p_phi_kernel']
        kernels['p2p_acc_kernel'] = all_kernels['cl_lib64_p2p_acc_kernel']
        kernels['p2p_pnacc_kernel'] = all_kernels['c_lib64_p2p_pnacc_kernel']
    elif device == 'cpu':
        kernels['p2p_phi_kernel'] = all_kernels['c_lib64_p2p_phi_kernel']
        kernels['p2p_acc_kernel'] = all_kernels['c_lib64_p2p_acc_kernel']
        kernels['p2p_pnacc_kernel'] = all_kernels['c_lib64_p2p_pnacc_kernel']
    else:
        print('Unsupported device...')
        sys.exit()

    return kernels, all_kernels


KERNELS, ALL_KERNELS = build_kernels()

kernel_library = NewExtensions(dtype='d', junroll=8)
kernel_library.build_kernels(device='gpu')


########## end of file ##########
