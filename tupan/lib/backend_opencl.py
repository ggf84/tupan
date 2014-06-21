# -*- coding: utf-8 -*-
#

"""This module implements the OpenCL backend to call CL-extensions.

"""


from __future__ import division
import os
import logging
import pyopencl as cl
from collections import namedtuple
from .utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'src')

CTX = cl.create_some_context()
DEV = CTX.devices[0]

UNROLL = 4

LSIZE = {}
LSIZE['fp32'] = 64
LSIZE['fp64'] = 64

VW = {}
VW['fp32'] = DEV.preferred_vector_width_float
VW['fp64'] = DEV.preferred_vector_width_double

FAST_LOCAL_MEM = True


@timings
def make_lib(fpwidth):
    """

    """
    cint = 'int' if fpwidth == 'fp32' else 'long'
    creal = 'float' if fpwidth == 'fp32' else 'double'
    LOGGER.debug("Building/Loading '%s' CL extension module.", fpwidth)

    fnames = ('phi_kernel.cl',
              'acc_kernel.cl',
              'acc_jerk_kernel.cl',
              'snap_crackle_kernel.cl',
              'tstep_kernel.cl',
              'pnacc_kernel.cl',
              'nreg_kernels.cl',
              'sakura_kernel.cl', )

    sources = []
    for fname in fnames:
        with open(os.path.join(PATH, fname), 'r') as fobj:
            sources.append(fobj.read())
    src = '\n\n'.join(sources)

    # setting options
    options = ' -I {path}'.format(path=PATH)
    options += ' -D CONFIG_USE_OPENCL'
    if fpwidth == 'fp64':
        options += ' -D CONFIG_USE_DOUBLE'
    if FAST_LOCAL_MEM:
        options += ' -D FAST_LOCAL_MEM'
    options += ' -D UNROLL={}'.format(UNROLL)
    options += ' -D LSIZE={}'.format(LSIZE[fpwidth])
    options += ' -D VW={}'.format(VW[fpwidth])
    options += ' -cl-fast-relaxed-math'
#    options += ' -cl-opt-disable'

    # building lib
    program = cl.Program(CTX, src)
    from ..config import get_cache_dir
    cllib = program.build(options=options, cache_dir=get_cache_dir())

    LOGGER.debug('CL extension module loaded: '
                 '(U)INT is (u)%s, REAL is %s.',
                 cint, creal)
    return cllib


LIB = {}
LIB['fp32'] = make_lib('fp32')
LIB['fp64'] = make_lib('fp64')


@bind_all(timings)
class CLKernel(object):

    def __init__(self, fpwidth, name):
        self.kernel = getattr(LIB[fpwidth], name)
        self.argtypes = None
        self._args = None

        lsize = LSIZE[fpwidth]
        gsize = lsize * 2  # XXX: lsize * DEV.max_compute_units ???
        self.global_size = (gsize, 1, 1)
        self.local_size = (lsize, 1, 1)

        self.queue = cl.CommandQueue(CTX)

        memf = cl.mem_flags
#        flags = memf.READ_WRITE | memf.USE_HOST_PTR
        flags = memf.READ_WRITE | memf.COPY_HOST_PTR

        from .utils.ctype import Ctype
        types = namedtuple('Types', ['c_int', 'c_int_p',
                                     'c_uint', 'c_uint_p',
                                     'c_real', 'c_real_p'])
        self.cty = types(
            c_int=vars(Ctype)['int'].type,
            c_int_p=lambda x: cl.Buffer(CTX, flags, hostbuf=x),
            c_uint=vars(Ctype)['uint'].type,
            c_uint_p=lambda x: cl.Buffer(CTX, flags, hostbuf=x),
            c_real=vars(Ctype)['real'].type,
            c_real_p=lambda x: cl.Buffer(CTX, flags, hostbuf=x),
            )

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        argtypes = self.argtypes
        self._args = [argtype(arg) for (argtype, arg) in zip(argtypes, args)]
        for (i, arg) in enumerate(self._args):
            self.kernel.set_arg(i, arg)

    def map_buffers(self, **kwargs):
        arrays = kwargs['outargs']
        buffers = self.args[len(kwargs['inpargs']):]

#        mapf = cl.map_flags
#        flags = mapf.READ | mapf.WRITE
#        queue = self.queue
#        for (ary, buf) in zip(arrays, buffers):
#            pointer, ev = cl.enqueue_map_buffer(
#                              queue,
#                              buf,
#                              flags,
#                              0,
#                              ary.shape,
#                              ary.dtype,
#                              'C'
#                          )
#            ev.wait()
        for (ary, buf) in zip(arrays, buffers):
            cl.enqueue_copy(self.queue, ary, buf)
        return arrays

    def run(self):
        cl.enqueue_nd_range_kernel(
            self.queue,
            self.kernel,
            self.global_size,
            self.local_size,
            ).wait()


# -- End of File --
