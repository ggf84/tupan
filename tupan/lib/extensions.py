# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import os
import logging
import numpy as np
from .utils import ctype
from .utils.timing import decallmethods, timings


logger = logging.getLogger(__name__)


try:
    import pyopencl as cl
    HAS_CL = True
except Exception as exc:
    cl = None
    HAS_CL = False
    logger.exception(str(exc))
    import warnings
    warnings.warn(
        """
        An Exception occurred when trying to import pyopencl module.
        See 'tupan.log' for more details.
        Continuing with C extension...
        """,
        stacklevel=1
    )


@decallmethods(timings)
class CLEnv(object):

    def __init__(self, prec, fast_math):
        self.prec = prec
        self.fast_math = fast_math
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)


@decallmethods(timings)
class CLKernel(object):

    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel
        self._gsize = None

    @property
    def global_size(self):
        return self._gsize

    @global_size.setter
    def global_size(self, ni):
        gs0 = ((ni-1)//2 + 1) * 2
        self._gsize = (gs0,)

    def alloc_local_memory(self, wgsize):
        itemsize = np.dtype(ctype.REAL).itemsize
        size = itemsize * wgsize
        return cl.LocalMemory(size)

    def set_local_memory(self, arg):
        return arg

    def set_int(self, arg):
        return ctype.UINT(arg)

    def set_float(self, arg):
        return ctype.REAL(arg)

    def set_array(self, arr):
        memf = cl.mem_flags
        return cl.Buffer(self.env.ctx,
                         memf.READ_WRITE | memf.USE_HOST_PTR,
                         hostbuf=arr)

    def set_arg(self, arg):
        if isinstance(arg, int):
            return self.set_int(arg)
        if isinstance(arg, float):
            return self.set_float(arg)
        if isinstance(arg, np.ndarray):
            return self.set_array(arg)
        if isinstance(arg, cl.LocalMemory):
            return self.set_local_memory(arg)

    def set_args(self, *args):
        self.dev_args = []
        for i, arg in enumerate(args):
            arg = self.set_arg(arg)
            self.dev_args.append(arg)
            self.kernel.set_arg(i, arg)

    def map_buffer(self, i, arr):
        mapf = cl.map_flags
        (pointer, ev) = cl.enqueue_map_buffer(self.env.queue,
                                              self.dev_args[i],
                                              mapf.READ,
                                              0,
                                              arr.shape,
                                              arr.dtype,
                                              "C")
        ev.wait()
#        cl.enqueue_copy(self.env.queue, arr, self.dev_args[i])

    def run(self):
        cl.enqueue_nd_range_kernel(self.env.queue,
                                   self.kernel,
                                   self.global_size,
                                   None,
                                   ).wait()


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

    def build(self):
        logger.debug("Building %s precision CL extension module.",
                     self.env.prec)

        # setting options
        options = " -I {path}".format(path=self.path)
        if self.env.prec is "double":
            options += " -D DOUBLE"
        if self.env.fast_math:
            options += " -cl-fast-relaxed-math"
#            options += " -cl-opt-disable"

        # building program
        self.program = cl.Program(
            self.env.ctx, self.src).build(options=options)

        logger.debug("done.")
        return self

    def __getattr__(self, name):
        return CLKernel(self.env, getattr(self.program, name))


@decallmethods(timings)
class CEnv(object):

    def __init__(self, prec, fast_math):
        self.prec = prec
        self.fast_math = fast_math


@decallmethods(timings)
class CKernel(object):

    def __init__(self, env, ffi, kernel):
        self.env = env
        self.ffi = ffi
        self.kernel = kernel

    def alloc_local_memory(self, wgsize):
        return None

    def set_int(self, arg):
        return arg

    def set_float(self, arg):
        return arg

    def set_array(self, arg):
        return self.ffi.cast("REAL *", arg.__array_interface__['data'][0])

    def set_arg(self, arg):
        if isinstance(arg, int):
            return self.set_int(arg)
        if isinstance(arg, float):
            return self.set_float(arg)
        if isinstance(arg, np.ndarray):
            return self.set_array(arg)

    def set_args(self, *args):
        self.ref_keep = []
        self.dev_args = []
        for i, arg in enumerate(args):
            if arg is not None:
                self.ref_keep.append(arg)
                arg = self.set_arg(arg)
                self.dev_args.append(arg)

    def map_buffer(self, i, arr):
        pass

    def run(self):
        args = self.dev_args
        self.kernel(*args)


@decallmethods(timings)
class CModule(object):

    def __init__(self, env):
        self.env = env

    def build(self):
        logger.debug("Building %s precision C extension module.",
                     self.env.prec)

        if self.env.prec is "double":
            from .cffi_wrap import clib_dp as program
            from .cffi_wrap import ffi_dp as ffi
        else:
            from .cffi_wrap import clib_sp as program
            from .cffi_wrap import ffi_sp as ffi
        self.program = program
        self.ffi = ffi

        logger.debug("done.")
        return self

    def __getattr__(self, name):
        return CKernel(self.env, self.ffi, getattr(self.program, name))


def get_kernel(name, exttype, prec):
    if not HAS_CL and exttype == "cl":
        exttype = "c"
    logger.debug(
        "Using '%s' from %s precision %s extension module.",
        name, ctype.prec, exttype.upper()
    )
    if exttype == "c":
        return getattr(
            CModule(CEnv(prec=prec, fast_math=True)).build(),
            name
        )
    elif exttype == "cl":
        return getattr(
            CLModule(CLEnv(prec=prec, fast_math=True)).build(),
            name
        )
    else:
        raise ValueError(
            "Inappropriate 'exttype' value. Supported values: ['c', 'cl']"
        )


########## end of file ##########
