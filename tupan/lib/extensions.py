# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import os
import ctypes
import logging
from fractions import gcd
from functools import partial
from collections import namedtuple
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
        stacklevel=1)


DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")


@decallmethods(timings)
class CLEnv(object):

    def __init__(self, prec, fast_math):
        self.prec = prec
        self.fast_math = fast_math
        self.ctx = cl.create_some_context()


@decallmethods(timings)
class CLKernel(object):

    def __init__(self, env, kernel):
        self.env = env
        self.kernel = kernel
        self.queue = cl.CommandQueue(self.env.ctx)
        self._gsize = None

        memf = cl.mem_flags
#        flags = memf.READ_WRITE | memf.USE_HOST_PTR
        flags = memf.READ_WRITE | memf.COPY_HOST_PTR
        clBuffer = partial(cl.Buffer, self.env.ctx, flags)

        types = namedtuple("Types", ["c_uint", "c_uint_p",
                                     "c_real", "c_real_p"])
        self.cty = types(c_uint=ctype.UINT,
                         c_uint_p=lambda x: clBuffer(hostbuf=x),
                         c_real=ctype.REAL,
                         c_real_p=lambda x: clBuffer(hostbuf=x),
                         )

    @property
    def global_size(self):
        return self._gsize

    @global_size.setter
    def global_size(self, ni):
        gs0 = ((ni-1)//2 + 1) * 2
        self._gsize = (gs0,)

        gs1 = ((ni-1)//self.wgsize + 1) * 2
        ls0 = gs0 // gs1

        self.local_size = (ls0,)

    def allocate_local_memory(self, nbufs, dtype):
        itemsize = dtype.itemsize
        dev = self.env.ctx.devices[0]

        size0 = dev.local_mem_size // itemsize
        size1 = dev.max_work_group_size
        wgsize = gcd(size0, size1 * nbufs) // nbufs
        self.wgsize = wgsize
        lmsize = wgsize * itemsize

        lmem = [cl.LocalMemory(lmsize) for i in range(nbufs)]
        self.lmem = lmem    # keep alive!
        return lmem

    def set_args(self, args, start=0):
        for (i, arg) in enumerate(args, start):
            self.kernel.set_arg(i, arg)

    def map_buffers(self, arrays, buffers):
#        mapf = cl.map_flags
#        flags = mapf.READ | mapf.WRITE
#        queue = self.queue
#        for (ary, buf) in zip(arrays, buffers):
#            (pointer, ev) = cl.enqueue_map_buffer(queue,
#                                                  buf,
#                                                  flags,
#                                                  0,
#                                                  ary.shape,
#                                                  ary.dtype,
#                                                  "C")
#            ev.wait()
        for (ary, buf) in zip(arrays, buffers):
            cl.enqueue_copy(self.queue, ary, buf)
        return arrays

    def run(self):
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.kernel,
                                   self.global_size,
                                   self.local_size,
                                   ).wait()


@decallmethods(timings)
class CLModule(object):

    def __init__(self, env):
        self.env = env

        files = ("smoothing.c",
                 "universal_kepler_solver.c",
                 #
                 "phi_kernel.cl",
                 "acc_kernel.cl",
                 "acc_jerk_kernel.cl",
                 "snap_crackle_kernel.cl",
                 "tstep_kernel.cl",
                 "pnacc_kernel.cl",
                 "nreg_kernels.cl",
                 "sakura_kernel.cl",
                 )

        sources = []
        for file in files:
            fname = os.path.join(PATH, file)
            with open(fname, 'r') as fobj:
                sources.append(fobj.read())
        self.src = "\n\n".join(sources)

    def build(self):
        logger.debug("Building %s precision CL extension module.",
                     self.env.prec)

        # setting options
        options = " -I {path}".format(path=PATH)
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
        self.kernel = kernel

        from_buffer = ctypes.c_char.from_buffer
        addressof = ctypes.addressof
        icast = partial(ffi.cast, "UINT *")
        rcast = partial(ffi.cast, "REAL *")

        types = namedtuple("Types", ["c_uint", "c_uint_p",
                                     "c_real", "c_real_p"])
        self.cty = types(c_uint=lambda x: x,
                         c_uint_p=lambda x: icast(addressof(from_buffer(x))),
                         c_real=lambda x: x,
                         c_real_p=lambda x: rcast(addressof(from_buffer(x))),
                         )

    def allocate_local_memory(self, numbufs, dtype):
        return []

    def set_args(self, args, start=0):
        self.args = args

    def map_buffers(self, arrays, buffers):
        return arrays

    def run(self):
        self.kernel(*self.args)


@decallmethods(timings)
class CModule(object):

    def __init__(self, env):
        self.env = env

    def build(self):
        logger.debug("Building %s precision C extension module.",
                     self.env.prec)

        fptype = "float" if self.env.prec == "single" else "double"

        from .cffi_wrap import wrap_lib
        self.ffi, self.program = wrap_lib(fptype)

        logger.debug("done.")
        return self

    def __getattr__(self, name):
        return CKernel(self.env, self.ffi, getattr(self.program, name))


@timings
def get_kernel(name, exttype, prec):
    if not HAS_CL and exttype == "CL":
        exttype = "C"
    logger.debug(
        "Using '%s' from %s precision %s extension module.",
        name, ctype.prec, exttype
    )
    if exttype == "C":
        return getattr(
            CModule(CEnv(prec=prec, fast_math=True)).build(),
            name
        )
    elif exttype == "CL":
        return getattr(
            CLModule(CLEnv(prec=prec, fast_math=True)).build(),
            name
        )
    else:
        raise ValueError(
            "Inappropriate 'exttype' value. Supported values: ['C', 'CL']"
        )


########## end of file ##########
