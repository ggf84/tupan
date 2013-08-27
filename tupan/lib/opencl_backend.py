# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import os
import logging
import pyopencl as cl
from fractions import gcd
from functools import partial
from collections import namedtuple
from .utils import ctype


logger = logging.getLogger(__name__)


DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")


def get_lib(env):
    """

    """
    logger.debug("Building/Loading %s precision CL extension module...",
                 env.prec)

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
    src = "\n\n".join(sources)

    # setting options
    options = " -I {path}".format(path=PATH)
    if env.fptype == "double":
        options += " -D DOUBLE"
    options += " -cl-fast-relaxed-math"
#    options += " -cl-opt-disable"

    # building lib
    env.ctx = cl.create_some_context()
    program = cl.Program(env.ctx, src)
    cllib = program.build(options=options)

    logger.debug("done.")
    return cllib


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


########## end of file ##########
