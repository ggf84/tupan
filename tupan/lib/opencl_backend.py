# -*- coding: utf-8 -*-
#

"""This module implements the OpenCL backend to call CL-extensions.

"""


from __future__ import division
import os
import sys
import math
import logging
import getpass
import pyopencl as cl
from functools import partial
from collections import namedtuple
from .utils import ctype


logger = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")

CACHE_DIR = os.path.join(os.path.expanduser('~'),
                         ".tupan",
                         "pyopencl-cache-uid{0}-py{1}".format(
                             getpass.getuser(),
                             ".".join(str(i) for i in sys.version_info)))

ctx = cl.create_some_context()


def make_lib(fptype):
    """

    """
    prec = "single" if fptype == 'float' else "double"
    logger.debug("Building/Loading %s precision CL extension module...",
                 prec)

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
    if fptype == "double":
        options += " -D DOUBLE"
    options += " -cl-fast-relaxed-math"
#    options += " -cl-opt-disable"

    # building lib
    program = cl.Program(ctx, src)
    cllib = program.build(options=options, cache_dir=CACHE_DIR)

    logger.debug("done.")
    return cllib


lib = {}
lib['single'] = make_lib('float')
lib['double'] = make_lib('double')


class CLKernel(object):

    def __init__(self, prec, name):
        self.kernel = getattr(lib[prec], name)
        self.queue = cl.CommandQueue(ctx)
        self._gsize = None

        memf = cl.mem_flags
#        flags = memf.READ_WRITE | memf.USE_HOST_PTR
        flags = memf.READ_WRITE | memf.COPY_HOST_PTR
        clBuffer = partial(cl.Buffer, ctx, flags)

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
        dev = ctx.devices[0]

        size0 = dev.local_mem_size // (itemsize * nbufs)
        size1 = dev.max_work_group_size
        size2 = 2**int(math.log(size0, 2))
        wgsize = min(size1, size2)
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
