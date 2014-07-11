# -*- coding: utf-8 -*-
#

"""This module implements the OpenCL backend to call CL-extensions.

"""


from __future__ import print_function, division
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
        self.inptypes = None
        self.outtypes = None

        kwginfo = cl.kernel_work_group_info
        LOGGER.debug("CL '%s' info: %s %s %s %s %s",
                     name,
                     self.kernel.get_work_group_info(
                         kwginfo.COMPILE_WORK_GROUP_SIZE, DEV),
                     self.kernel.get_work_group_info(
                         kwginfo.LOCAL_MEM_SIZE, DEV),
                     self.kernel.get_work_group_info(
                         kwginfo.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, DEV),
                     self.kernel.get_work_group_info(
                         kwginfo.PRIVATE_MEM_SIZE, DEV),
                     self.kernel.get_work_group_info(
                         kwginfo.WORK_GROUP_SIZE, DEV),
                     )

        self.vector_width = VW[fpwidth]
        self.local_size = (LSIZE[fpwidth], 1, 1)
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

    def set_args(self, inpargs, outargs):
        bufs = []

        # set inpargs
        self.inp_argbuf = []
        for (i, argtype) in enumerate(self.inptypes):
            arg = inpargs[i]
            buf = argtype(arg)
            self.inp_argbuf.append((arg, buf))
            bufs.append(buf)

        # set outargs
        self.out_argbuf = []
        for (i, argtype) in enumerate(self.outtypes):
            arg = outargs[i]
            buf = argtype(arg)
            self.out_argbuf.append((arg, buf))
            bufs.append(buf)

        for (i, buf) in enumerate(bufs):
            self.kernel.set_arg(i, buf)

        ni = inpargs[0]
        vw = self.vector_width
        lsize = self.local_size[0]
        n = (ni + vw - 1) // vw
        ngroups = (n + lsize - 1) // lsize
        ngroups = max(ngroups, DEV.max_compute_units)
        self.global_size = (lsize * ngroups, 1, 1)

    def map_buffers(self):
        for (arg, buf) in self.out_argbuf:
            cl.enqueue_copy(self.queue, arg, buf)
        return [arg for (arg, buf) in self.out_argbuf]

#        mapf = cl.map_flags
#        flags = mapf.READ | mapf.WRITE
#        queue = self.queue
#        for (arg, buf) in self.out_argbuf:
#            arg[...], ev = cl.enqueue_map_buffer(
#                queue,
#                buf,
#                flags,
#                0,
#                arg.shape,
#                arg.dtype,
#                'C'
#                )
#            ev.wait()
#        return [arg for (arg, buf) in self.out_argbuf]

    def run(self):
        cl.enqueue_nd_range_kernel(
            self.queue,
            self.kernel,
            self.global_size,
            self.local_size,
            ).wait()


# -- End of File --
