# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import os
import logging
from functools import reduce
from collections import OrderedDict
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
        self.dev_buff = {}
        self._lsize = None
        self._lsize_max = None
        self._gsize = None

    @property
    def local_size(self):
        return self._lsize

    @local_size.setter
    def local_size(self, lsize):
        self._lsize = (lsize,)
        self._lsize_max = (lsize,)

    @property
    def global_size(self):
        return self._gsize

    @global_size.setter
    def global_size(self, ni):
        gs0 = ((ni-1)//2 + 1) * 2
        self._gsize = (gs0,)

    def set_local_memory(self, i, arg):
        def foo(x, y):
            return x * y
#        size = np.dtype(ctype.REAL).itemsize * reduce(foo, self.local_size)
#        size = np.dtype(ctype.REAL).itemsize * reduce(foo, (self._lsize_max,))
        size = 8 * reduce(foo, self._lsize_max)
        arg = cl.LocalMemory(size * arg)
        self.kernel.set_arg(i, arg)

    def set_int(self, i, arg):
        arg = ctype.UINT(arg)
        self.kernel.set_arg(i, arg)

    def set_float(self, i, arg):
        arg = ctype.REAL(arg)
        self.kernel.set_arg(i, arg)

    def set_array(self, i, arr):
        memf = cl.mem_flags
        self.dev_buff[i] = cl.Buffer(self.env.ctx,
                                     memf.READ_WRITE | memf.COPY_HOST_PTR,
                                     hostbuf=arr)

        arg = self.dev_buff[i]
        self.kernel.set_arg(i, arg)

    def allocate_buffer(self, i, shape, dtype):
        memf = cl.mem_flags
        arr = np.zeros(shape, dtype=dtype)
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
        (pointer, ev) = cl.enqueue_map_buffer(self.env.queue,
                                              self.dev_buff[i],
                                              mapf.READ,
                                              0,
                                              arr.shape,
                                              arr.dtype,
                                              "C")
        ev.wait()

#        cl.enqueue_copy(self.env.queue, arr, self.dev_buff[i])

    def run(self):
        cl.enqueue_nd_range_kernel(self.env.queue,
                                   self.kernel,
                                   self.global_size,
                                   None,
                                   ).wait()

    def set_arg(self, i, arg):
        if isinstance(arg, int):
            self.set_int(i, arg)
        if isinstance(arg, float):
            self.set_float(i, arg)
        if isinstance(arg, np.ndarray):
            self.set_array(i, arg)

    def set_args(self, *args):
        for i, arg in enumerate(args):
            self.set_arg(i, arg)


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
        self.dev_args[i] = self.ffi.cast(
            "REAL *", arg.__array_interface__['data'][0]
        )

    def allocate_buffer(self, i, shape, dtype):
        arg = np.zeros(shape, dtype=dtype)
        self.dev_args[i] = self.ffi.cast(
            "REAL *", arg.__array_interface__['data'][0]
        )
        return arg

    def map_buffer(self, i, arr):
        pass

    def run(self):
        args = self.dev_args.values()
        self.kernel(*args)

    def set_arg(self, i, arg):
        if isinstance(arg, int):
            self.set_int(i, arg)
        if isinstance(arg, float):
            self.set_float(i, arg)
        if isinstance(arg, np.ndarray):
            self.set_array(i, arg)

    def set_args(self, *args):
        for i, arg in enumerate(args):
            self.set_arg(i, arg)


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
