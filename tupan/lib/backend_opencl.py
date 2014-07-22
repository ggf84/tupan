# -*- coding: utf-8 -*-
#

"""This module implements the OpenCL backend to call CL-extensions.

"""


import os
import logging
import pyopencl as cl
from tupan.config import options
from .utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'src')


@bind_all(timings)
class CLDriver(object):
    """

    """
    def __init__(self, fpwidth=options.fpwidth):
        self.ctx = cl.create_some_context()
        self.dev = self.ctx.devices[0]
        self.queue = cl.CommandQueue(self.ctx)

        self.fpwidth = fpwidth
        self.unroll = 4
        self.lsize = 64
        self.fast_local_mem = True

        self.vw = self.dev.preferred_vector_width_float
        if self.fpwidth == 'fp64':
            self.vw = self.dev.preferred_vector_width_double

        self._make_lib()

    def _make_lib(self):
        cint = 'int' if self.fpwidth == 'fp32' else 'long'
        creal = 'float' if self.fpwidth == 'fp32' else 'double'
        LOGGER.debug("Building '%s' CL extension module...", self.fpwidth)

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
        if self.fpwidth == 'fp64':
            options += ' -D CONFIG_USE_DOUBLE'
        if self.fast_local_mem:
            options += ' -D FAST_LOCAL_MEM'
        options += ' -D UNROLL={}'.format(self.unroll)
        options += ' -D LSIZE={}'.format(self.lsize)
        options += ' -D VW={}'.format(self.vw)
        options += ' -cl-fast-relaxed-math'
#        options += ' -cl-opt-disable'

        # building lib
        program = cl.Program(self.ctx, src)
        from ..config import get_cache_dir
        self.lib = program.build(options=options, cache_dir=get_cache_dir())

        LOGGER.debug('CL extension module loaded: '
                     '(U)INT is (u)%s, REAL is %s.',
                     cint, creal)

        mf = cl.mem_flags

#        def iptr(arg, flags=mf.READ_ONLY | mf.USE_HOST_PTR):
#            return cl.Buffer(self.ctx, flags, hostbuf=arg)
#
#        def optr(arg, flags=mf.WRITE_ONLY | mf.USE_HOST_PTR):
#            return cl.Buffer(self.ctx, flags, hostbuf=arg)

        def iptr(arg, flags=mf.READ_ONLY | mf.COPY_HOST_PTR):
            return cl.Buffer(self.ctx, flags, hostbuf=arg)

#        def optr(arg, flags=mf.WRITE_ONLY | mf.COPY_HOST_PTR):
#            return cl.Buffer(self.ctx, flags, hostbuf=arg)

#        def iptr(arg, flags=mf.READ_ONLY | mf.ALLOC_HOST_PTR):
#            buf = cl.Buffer(self.ctx, flags, size=arg.nbytes)
#            cl.enqueue_copy(self.queue, buf, arg)
#            return buf

        def optr(arg, flags=mf.WRITE_ONLY | mf.ALLOC_HOST_PTR):
            return cl.Buffer(self.ctx, flags, size=arg.nbytes)

        from .utils.ctype import Ctype

        self.c_int = vars(Ctype)['int'].type
        self.c_uint = vars(Ctype)['uint'].type
        self.c_real = vars(Ctype)['real'].type
        self.iptr = iptr
        self.optr = optr

    def get_kernel(self, name):
        LOGGER.debug("Using '%s' function from 'CL' backend.", name)
        kernel = getattr(self.lib, name)
        kwginfo = cl.kernel_work_group_info
        LOGGER.debug("CL '%s' info: %s %s %s %s %s",
                     name,
                     kernel.get_work_group_info(
                         kwginfo.COMPILE_WORK_GROUP_SIZE, self.dev),
                     kernel.get_work_group_info(
                         kwginfo.LOCAL_MEM_SIZE, self.dev),
                     kernel.get_work_group_info(
                         kwginfo.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.dev),
                     kernel.get_work_group_info(
                         kwginfo.PRIVATE_MEM_SIZE, self.dev),
                     kernel.get_work_group_info(
                         kwginfo.WORK_GROUP_SIZE, self.dev),
                     )
        return CLKernel(kernel)


drv = CLDriver()


@bind_all(timings)
class CLKernel(object):
    """

    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.inptypes = None
        self.outtypes = None
        self.iarg = {}
        self.ibuf = {}
        self.oarg = {}
        self.obuf = {}

        self.vector_width = drv.vw
        self.local_size = (drv.lsize, 1, 1)
        self.queue = drv.queue
        self.min_ngroups = drv.dev.max_compute_units

    def set_args(self, inpargs, outargs):
        bufs = []

        if self.inptypes is None:
            types = []
            for arg in inpargs:
                if isinstance(arg, int):
                    types.append(drv.c_int if arg < 0 else drv.c_uint)
                elif isinstance(arg, float):
                    types.append(drv.c_real)
                else:
                    types.append(drv.iptr)
            self.inptypes = types

        if self.outtypes is None:
            self.outtypes = [drv.optr for _ in outargs]

        # set inpargs
        for (i, argtype) in enumerate(self.inptypes):
            arg = inpargs[i]
            buf = argtype(arg)
            self.iarg[i] = arg
            self.ibuf[i] = buf
            bufs.append(buf)

        # set outargs
        for (i, argtype) in enumerate(self.outtypes):
            arg = outargs[i]
            buf = argtype(arg)
            self.oarg[i] = arg
            self.obuf[i] = buf
            bufs.append(buf)

        self.bufs = bufs

        ni = inpargs[0]
        vw = self.vector_width
        lsize = self.local_size[0]
        n = (ni + vw - 1) // vw
        ngroups = (n + lsize - 1) // lsize
        ngroups = max(ngroups, self.min_ngroups)
        self.global_size = (lsize * ngroups, 1, 1)

        for (i, buf) in enumerate(self.bufs):
            self.kernel.set_arg(i, buf)

    def map_buffers(self):
        for (key, arg) in self.oarg.items():
            buf = self.obuf[key]
            cl.enqueue_copy(self.queue, arg, buf)
        return list(self.oarg.values())

#        flags = cl.map_flags.READ
#        queue = self.queue
#        for (key, arg) in self.oarg.items():
#            buf = self.obuf[key]
#            arg[...], ev = cl.enqueue_map_buffer(
#                queue,
#                buf,
#                flags,
#                0,
#                arg.shape,
#                arg.dtype,
#                )
#            ev.wait()
#        return list(self.oarg.values())

    def run(self):
        cl.enqueue_nd_range_kernel(
            self.queue,
            self.kernel,
            self.global_size,
            self.local_size,
            ).wait()


# -- End of File --
