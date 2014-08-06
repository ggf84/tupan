# -*- coding: utf-8 -*-
#

"""This module implements the OpenCL backend to call CL-extensions.

"""


import os
import logging
import pyopencl as cl
from functools import partial
from collections import defaultdict
from ..config import options, get_cache_dir
from .utils.ctype import Ctype
from .utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'src')


@bind_all(timings)
class Context(object):
    """

    """
    def __init__(self, cl_platform):
        cl_devices = cl_platform.get_devices()

#        # emulate multiple devices
#        cl_devices = cl_devices[0].create_sub_devices(
#            [cl.device_partition_property.EQUALLY, 1])
#        cl_devices = cl_devices[0].create_sub_devices(
#            [cl.device_partition_property.BY_COUNTS, 1, 1, 1])

        self.cl_context = cl.Context(devices=cl_devices)
        self.queue = Queue(self.cl_context)
        self.devices = [Device(self.cl_context, cl_device)
                        for cl_device in cl_devices]

        self.ibuf = defaultdict(partial(cl.Buffer, self.cl_context,
                                        cl.mem_flags.READ_ONLY,
                                        size=1))

        self.obuf = defaultdict(partial(cl.Buffer, self.cl_context,
                                        cl.mem_flags.WRITE_ONLY,
                                        size=1))

        self.reset_buf_counts()

    def to_ibuf(self, ary,
                flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR):
        buf = cl.Buffer(self.cl_context, flags, hostbuf=ary)
        self.ibuf[self.ibuf_count] = buf
        self.ibuf_count += 1
        return buf

    def to_obuf(self, ary,
                flags=cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR):
        buf = cl.Buffer(self.cl_context, flags, hostbuf=ary)
        self.obuf[self.obuf_count] = buf
        self.obuf_count += 1
        return buf

    def get_ibuf(self, ary, flags=cl.mem_flags.READ_ONLY):
        size = ary.nbytes
        buf = self.ibuf[self.ibuf_count]
        if size > buf.size:
            buf = cl.Buffer(self.cl_context, flags, size=size)
            self.ibuf[self.ibuf_count] = buf
        self.ibuf_count += 1
        return buf

    def get_obuf(self, ary, flags=cl.mem_flags.WRITE_ONLY):
        size = ary.nbytes
        buf = self.obuf[self.obuf_count]
        if size > buf.size:
            buf = cl.Buffer(self.cl_context, flags, size=size)
            self.obuf[self.obuf_count] = buf
        self.obuf_count += 1
        return buf

    def reset_buf_counts(self):
        self.ibuf_count = 0
        self.obuf_count = 0


@bind_all(timings)
class Queue(object):
    """

    """
    events = []

    def __init__(self, cl_context, cl_device=None,
                 out_of_order=True, profiling=False):
        properties = 0
        cqp = cl.command_queue_properties
        if out_of_order:
            properties |= cqp.OUT_OF_ORDER_EXEC_MODE_ENABLE
        if profiling:
            properties |= cqp.PROFILING_ENABLE
        self.cl_queue = cl.CommandQueue(cl_context,
                                        device=cl_device,
                                        properties=properties)

    def wait_for_events(self):
        cl.wait_for_events(self.events)
        del self.events[:]

    def enqueue_read_buffer(self, buf, ary,
                            device_offset=0,
                            wait_for=None,
                            is_blocking=False):
        event = cl.enqueue_copy(self.cl_queue, ary, buf,
                                device_offset=device_offset,
                                wait_for=wait_for,
                                is_blocking=is_blocking)
        self.cl_queue.flush()
        self.events.append(event)

    def enqueue_write_buffer(self, buf, ary,
                             device_offset=0,
                             wait_for=None,
                             is_blocking=False):
        event = cl.enqueue_copy(self.cl_queue, buf, ary,
                                device_offset=device_offset,
                                wait_for=wait_for,
                                is_blocking=is_blocking)
        self.cl_queue.flush()
        self.events.append(event)

    def enqueue_nd_range_kernel(self, kernel,
                                global_work_size,
                                local_work_size,
                                global_work_offset,
                                wait_for=None):
        event = cl.enqueue_nd_range_kernel(self.cl_queue,
                                           kernel,
                                           global_work_size,
                                           local_work_size,
                                           global_work_offset,
                                           wait_for=wait_for)
        self.cl_queue.flush()
        self.events.append(event)


@bind_all(timings)
class Device(object):
    """

    """
    idx = 0

    def __init__(self, cl_context, cl_device):
        self.cl_context = cl_context
        self.cl_device = cl_device
        self.queue = Queue(cl_context, cl_device)
        self.program = Program(cl_context, cl_device).build()

        self.print_info()

    def print_info(self, short=True):
        i = type(self).idx
        indent = ' ' * 4
        print(indent + str([i]) + ' ' + str(self.cl_device))
        if not short:
            for name in sorted(dir(cl.device_info)):
                if not name.startswith('_') and name != 'to_string':
                    try:
                        info = getattr(cl.device_info, name)
                        value = self.cl_device.get_info(info)
                    except cl.LogicError:
                        value = '<error>'
                    msg = 2 * indent + '{}: {}'
                    print(msg.format(name, value))
        type(self).idx += 1


@bind_all(timings)
class Program(object):
    """

    """
    def __init__(self, cl_context, cl_device):
        fnames = ('phi_kernel.cl',
                  'acc_kernel.cl',
                  'acc_jerk_kernel.cl',
                  'snap_crackle_kernel.cl',
                  'tstep_kernel.cl',
                  'pnacc_kernel.cl',
                  'nreg_kernels.cl',
                  'sakura_kernel.cl', )

        fsources = []
        for fname in fnames:
            with open(os.path.join(PATH, fname), 'r') as fobj:
                fsources.append(fobj.read())
        source = '\n'.join(fsources)

        self.cl_context = cl_context
        self.cl_device = cl_device
        self.cl_program = cl.Program(cl_context, source)
        self.kernel = None

    def build(self, fpwidth=options.fpwidth):
        unroll = 4
        lsize = 64
        fast_local_mem = True
        vw = (self.cl_device.preferred_vector_width_float
              if fpwidth == 'fp32'
              else self.cl_device.preferred_vector_width_double)

        # setting program options
        opts = ' -D VW={}'.format(vw)
        opts += ' -D UNROLL={}'.format(unroll)
        opts += ' -D LSIZE={}'.format(lsize)
        opts += ' -D CONFIG_USE_OPENCL'
        if fpwidth == 'fp64':
            opts += ' -D CONFIG_USE_DOUBLE'
        if fast_local_mem:
            opts += ' -D FAST_LOCAL_MEM'
        opts += ' -cl-fast-relaxed-math'
#        opts += ' -cl-opt-disable'
        opts += ' -I {path}'.format(path=PATH)

        self.cl_program.build(options=opts,
                              devices=[self.cl_device],
                              cache_dir=get_cache_dir())

        kwgi = cl.kernel_work_group_info
        kernels = self.cl_program.all_kernels()
        for kernel in kernels:
            name = kernel.function_name
            kernel.vw = vw if name != 'sakura_kernel' else 1
            kernel.unroll = unroll
            kernel.lsize = lsize
            kernel.name = name
            LOGGER.debug(
                "CL '%s' info: %s %s %s %s %s on %s",
                name,
                kernel.get_work_group_info(
                    kwgi.COMPILE_WORK_GROUP_SIZE,
                    self.cl_device),
                kernel.get_work_group_info(
                    kwgi.LOCAL_MEM_SIZE,
                    self.cl_device),
                kernel.get_work_group_info(
                    kwgi.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                    self.cl_device),
                kernel.get_work_group_info(
                    kwgi.PRIVATE_MEM_SIZE,
                    self.cl_device),
                kernel.get_work_group_info(
                    kwgi.WORK_GROUP_SIZE,
                    self.cl_device),
                self.cl_device
            )

        self.kernel = {kernel.name: kernel for kernel in kernels}

#        for kernel in self.kernel.values():
#            print(kernel, kernel.attributes, kernel.name)
#        print('-'*48)

        return self


@bind_all(timings)
class Platform(object):
    """

    """
    def __init__(self, idx):
        cl_platforms = cl.get_platforms()
        for i, cl_platform in enumerate(cl_platforms):
            print(str([i]) + ' ' + str(cl_platform))

        self.cl_platform = cl_platforms[idx]
        self.context = Context(self.cl_platform)

    def get_kernel(self, name):
        return CLKernel(name)


drv = Platform(0)


@bind_all(timings)
class CLKernel(object):
    """

    """
    def __init__(self, name):
        self.name = name
        self.inptypes = None
        self.outtypes = None
        self.iarg = {}
        self.ibuf = {}
        self.oarg = {}
        self.obuf = {}

    def set_args(self, inpargs, outargs):
        bufs = []

        drv.context.reset_buf_counts()

        if self.inptypes is None:
            types = []
            for arg in inpargs:
                if isinstance(arg, int):
                    int_t = Ctype.int_t
                    uint_t = Ctype.uint_t
                    types.append(int_t if arg < 0 else uint_t)
                elif isinstance(arg, float):
                    real_t = Ctype.real_t
                    types.append(real_t)
                else:
                    iptr = drv.context.to_ibuf
                    types.append(iptr)
            self.inptypes = types

        if self.outtypes is None:
            optr = drv.context.get_obuf
            self.outtypes = [optr for _ in outargs]

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

    def map_buffers(self):
        for (key, arg) in self.oarg.items():
            buf = self.obuf[key]
            drv.context.queue.enqueue_read_buffer(buf, arg)
        drv.context.queue.wait_for_events()
        return list(self.oarg.values())

    def run(self):
        offset = 0
        voffset = 0
        ni = self.iarg[0]
        name = self.name
        uint_t = Ctype.uint_t
        ndevs = len(drv.context.devices)

        for device in drv.context.devices:
            kernel = device.program.kernel[name]

            vw = kernel.vw
            lsize = kernel.lsize

            n = (ni + vw - 1) // vw
            n_per_dev = (n + ndevs - 1) // ndevs

            ngroups = (n_per_dev + lsize - 1) // lsize
            gsize = lsize * ngroups
            offset = (voffset + vw - 1) // vw

            local_work_size = (lsize, 1, 1)
            global_work_size = (gsize, 1, 1)
            global_work_offset = (offset, 0, 0)

            offset += gsize
            voffset += gsize * vw
            nn = min(offset, n)

            kernel.set_arg(0, uint_t(nn))
            for (j, buf) in enumerate(self.bufs[1:], start=1):
                kernel.set_arg(j, buf)

            device.queue.enqueue_nd_range_kernel(
                kernel,
                global_work_size,
                local_work_size,
                global_work_offset)

        drv.context.queue.wait_for_events()


# -- End of File --
