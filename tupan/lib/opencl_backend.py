# -*- coding: utf-8 -*-
#

"""This module implements the OpenCL backend to call CL-extensions.

"""


from __future__ import division
import os
import sys
import logging
import getpass
import pyopencl as cl
from functools import partial
from collections import namedtuple


logger = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, "src")

CACHE_DIR = os.path.join(os.path.expanduser('~'),
                         ".tupan",
                         "pyopencl-cache-uid{0}-py{1}".format(
                             getpass.getuser(),
                             ".".join(str(i) for i in sys.version_info)))

ctx = cl.create_some_context()
dev = ctx.devices[0]

VECTOR_WIDTH = {}
VECTOR_WIDTH["float32"] = dev.preferred_vector_width_float
VECTOR_WIDTH["float64"] = dev.preferred_vector_width_double


def make_lib(prec):
    """

    """
    cint = "int" if prec == "float32" else "long"
    creal = "float" if prec == 'float32' else "double"
    logger.debug("Building/Loading %s CL extension module...",
                 prec)

    files = ("phi_kernel.cl",
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
    options += " -D CONFIG_USE_OPENCL"
    if prec == "float64":
        options += " -D CONFIG_USE_DOUBLE"
    options += " -D VECTOR_WIDTH={}".format(VECTOR_WIDTH[prec])
    options += " -cl-fast-relaxed-math"
#    options += " -cl-opt-disable"

    # building lib
    program = cl.Program(ctx, src)
    cllib = program.build(options=options, cache_dir=CACHE_DIR)

    logger.debug("done.")
    return cllib


lib = {}
lib['float32'] = make_lib('float32')
lib['float64'] = make_lib('float64')


class CLKernel(object):

    def __init__(self, prec, name):
        self.kernel = getattr(lib[prec], name)
        self.vector_width = VECTOR_WIDTH[prec]
        self.queue = cl.CommandQueue(ctx)

        memf = cl.mem_flags
#        flags = memf.READ_WRITE | memf.USE_HOST_PTR
        flags = memf.READ_WRITE | memf.COPY_HOST_PTR
        clBuffer = partial(cl.Buffer, ctx, flags)

        from .utils.ctype import ctypedict
        types = namedtuple("Types", ["c_int", "c_int_p",
                                     "c_uint", "c_uint_p",
                                     "c_real", "c_real_p"])
        self.cty = types(c_int=ctypedict["int"].type,
                         c_int_p=lambda x: clBuffer(hostbuf=x),
                         c_uint=ctypedict["uint"].type,
                         c_uint_p=lambda x: clBuffer(hostbuf=x),
                         c_real=ctypedict["real"].type,
                         c_real_p=lambda x: clBuffer(hostbuf=x),
                         )

    def set_gsize(self, gsize):
        gs = (gsize + self.vector_width - 1) // self.vector_width
        self.global_size = (gs, 1, 1)

#    @property
#    def global_size(self):
#        return self._gsize
#
#    @global_size.setter
#    def global_size(self, ni):
#        gs0 = ((ni-1)//2 + 1) * 2
#        self._gsize = (gs0,)
#
#        gs1 = ((ni-1)//self.wgsize + 1) * 2
#        ls0 = gs0 // gs1
#
#        self.local_size = (ls0,)
#
#    def allocate_local_memory(self, nbufs, sctype):
#        import math
#        from .utils.ctype import ctypedict
#        dtype = ctypedict[sctype]
#
#        itemsize = dtype.itemsize
#        dev = ctx.devices[0]
#
#        size0 = dev.local_mem_size // (itemsize * nbufs)
#        size1 = dev.max_work_group_size
#        size2 = 2**int(math.log(size0, 2))
#        wgsize = min(size1, size2)
#        self.wgsize = wgsize
#        lmsize = wgsize * itemsize
#
#        lmem = [cl.LocalMemory(lmsize) for i in range(nbufs)]
#        self.lmem = lmem    # keep alive!
#        return lmem

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
                                   None,
                                   ).wait()


########## end of file ##########
