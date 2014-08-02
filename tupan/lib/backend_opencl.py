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


class Context(object):
    """

    """
    def __init__(self, devices):
        self.ctx = cl.Context(devices=devices)
        self.devices = devices

        self.ibuf = defaultdict(partial(cl.Buffer, self.ctx,
                                        cl.mem_flags.READ_ONLY,
                                        size=1))

        self.obuf = defaultdict(partial(cl.Buffer, self.ctx,
                                        cl.mem_flags.WRITE_ONLY,
                                        size=1))

        self.reset_buf_counts()

    def to_ibuf(self, ary,
                flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR):
        buf = cl.Buffer(self.ctx, flags, hostbuf=ary)
        self.ibuf[self.ibuf_count] = buf
        self.ibuf_count += 1
        return buf

    def to_obuf(self, ary,
                flags=cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR):
        buf = cl.Buffer(self.ctx, flags, hostbuf=ary)
        self.obuf[self.obuf_count] = buf
        self.obuf_count += 1
        return buf

    def get_ibuf(self, ary, flags=cl.mem_flags.READ_ONLY):
        size = ary.nbytes
        buf = self.ibuf[self.ibuf_count]
        if size > buf.size:
            buf = cl.Buffer(self.ctx, flags, size=size)
            self.ibuf[self.ibuf_count] = buf
        self.ibuf_count += 1
        return buf

    def get_obuf(self, ary, flags=cl.mem_flags.WRITE_ONLY):
        size = ary.nbytes
        buf = self.obuf[self.obuf_count]
        if size > buf.size:
            buf = cl.Buffer(self.ctx, flags, size=size)
            self.obuf[self.obuf_count] = buf
        self.obuf_count += 1
        return buf

    def reset_buf_counts(self):
        self.ibuf_count = 0
        self.obuf_count = 0

    def create_command_queue(self, out_of_order=True, profiling=False):
        properties = 0
        cqp = cl.command_queue_properties
        if out_of_order:
            properties |= cqp.OUT_OF_ORDER_EXEC_MODE_ENABLE
        if profiling:
            properties |= cqp.PROFILING_ENABLE
        return CommandQueue(self.ctx, properties=properties)

    def create_program_from_source(self, fnames):
        fsources = []
        for fname in fnames:
            with open(os.path.join(PATH, fname), 'r') as fobj:
                fsources.append(fobj.read())
        source = '\n'.join(fsources)
        return Program(self.ctx, source)


class CommandQueue(object):
    """

    """
    def __init__(self, context, properties=0):
        self.queues = [cl.CommandQueue(context,
                                       device=device,
                                       properties=properties)
                       for device in context.devices]
        self.context = context
        self.events = []

    def flush(self):
        for queue in self.queues:
            queue.flush()

    def finish(self):
        for queue in self.queues:
            queue.finish()

    def wait_for_events(self):
        self.flush()
        cl.wait_for_events(self.events)
        del self.events[:]

    def enqueue_read_buffer(self, buf, ary,
                            nqueues=None,
                            device_offset=0,
                            wait_for=None,
                            is_blocking=False):
        events = [cl.enqueue_copy(queue, ary, buf,
                                  device_offset=device_offset,
                                  wait_for=wait_for,
                                  is_blocking=is_blocking)
                  for queue in self.queues[:nqueues]]
        self.events += events

#        events = [cl.enqueue_read_buffer(queue, buf, ary,
#                                         device_offset=device_offset,
#                                         wait_for=wait_for,
#                                         is_blocking=is_blocking)
#                  for queue in self.queues[:nqueues]]
#        self.events += events

    def enqueue_write_buffer(self, buf, ary,
                             nqueues=None,
                             device_offset=0,
                             wait_for=None,
                             is_blocking=False):
        events = [cl.enqueue_copy(queue, buf, ary,
                                  device_offset=device_offset,
                                  wait_for=wait_for,
                                  is_blocking=is_blocking)
                  for queue in self.queues[:nqueues]]
        self.events += events

#        events = [cl.enqueue_write_buffer(queue, buf, ary,
#                                          device_offset=device_offset,
#                                          wait_for=wait_for,
#                                          is_blocking=is_blocking)
#                  for queue in self.queues[:nqueues]]
#        self.events += events

    def enqueue_nd_range_kernel(self, kernels,
                                global_work_sizes,
                                local_work_sizes,
                                global_work_offsets,
                                wait_for=None):
        events = [cl.enqueue_nd_range_kernel(queue,
                                             kernels[i],
                                             global_work_sizes[i],
                                             local_work_sizes[i],
                                             global_work_offsets[i],
                                             wait_for=wait_for)
                  for i, queue in enumerate(self.queues)]
        self.events += events


class Program(object):
    """

    """
    def __init__(self, context, source):
        self.context = context
        self.source = source
        self.programs = []

    def build(self, fpwidth):
        unroll = 4
        lsize = 64
        fast_local_mem = True

        # setting program options
        popt = ' -D UNROLL={}'.format(unroll)
        popt += ' -D LSIZE={}'.format(lsize)
        popt += ' -D CONFIG_USE_OPENCL'
        if fpwidth == 'fp64':
            popt += ' -D CONFIG_USE_DOUBLE'
        if fast_local_mem:
            popt += ' -D FAST_LOCAL_MEM'
        popt += ' -cl-fast-relaxed-math'
#        popt += ' -cl-opt-disable'
        popt += ' -I {path}'.format(path=PATH)

        del self.programs[:]
        for device in self.context.devices:
            vw = (device.preferred_vector_width_float
                  if fpwidth == 'fp32'
                  else device.preferred_vector_width_double)

            # setting device-specific options
            dsopt = ' -D VW={}'.format(vw)

            options = dsopt + popt
            program = cl.Program(self.context, self.source)
            program.build(options=options,
                          devices=[device],
                          cache_dir=get_cache_dir())

            program.min_ngroups = device.max_compute_units
            program.device = device
            program.unroll = unroll
            program.lsize = lsize
            program.vw = vw

            self.programs.append(program)

    def create_kernel(self, name):
        kernels = []
        for program in self.programs:
            kernel = cl.Kernel(program, name)

            kernel.min_ngroups = program.min_ngroups
            kernel.unroll = program.unroll
            kernel.lsize = program.lsize
            kernel.vw = program.vw if name != 'sakura_kernel' else 1
            kernel.name = name

            kernels.append(kernel)

            kwgi = cl.kernel_work_group_info
            device = program.device
            LOGGER.debug(
                "CL '%s' info: %s %s %s %s %s",
                name,
                kernel.get_work_group_info(
                    kwgi.COMPILE_WORK_GROUP_SIZE,
                    device),
                kernel.get_work_group_info(
                    kwgi.LOCAL_MEM_SIZE,
                    device),
                kernel.get_work_group_info(
                    kwgi.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                    device),
                kernel.get_work_group_info(
                    kwgi.PRIVATE_MEM_SIZE,
                    device),
                kernel.get_work_group_info(
                    kwgi.WORK_GROUP_SIZE,
                    device),
            )

#        for kernel in kernels:
#            print(kernel, kernel.attributes, kernel.name)
#        print('-'*48)

        return kernels


@bind_all(timings)
class CLDriver(object):
    """

    """
    def __init__(self, fpwidth=options.fpwidth):
        self.fpwidth = fpwidth

        fnames = ('phi_kernel.cl',
                  'acc_kernel.cl',
                  'acc_jerk_kernel.cl',
                  'snap_crackle_kernel.cl',
                  'tstep_kernel.cl',
                  'pnacc_kernel.cl',
                  'nreg_kernels.cl',
                  'sakura_kernel.cl', )

        LOGGER.debug("Building CL driver in '%s' precision...", fpwidth)

        platforms = cl.get_platforms()
        for i, platform in enumerate(platforms):
            print(str([i]) + ' ' + str(platform))

        devices = platforms[0].get_devices()

#        # emulate multiple devices
#        devices = devices[0].create_sub_devices(
#            [cl.device_partition_property.EQUALLY, 1])
#        devices = devices[0].create_sub_devices(
#            [cl.device_partition_property.BY_COUNTS, 1, 1, 1])

        self.print_info(devices, short=True)

        self.context = Context(devices)
        self.queue = self.context.create_command_queue()
        self.program = self.context.create_program_from_source(fnames)
        self.program.build(fpwidth)

    def print_info(self, devices, short=True):
        indent = ' ' * 4
        for i, device in enumerate(devices):
            print(indent + str([i]) + ' ' + str(device))
            if not short:
                for name in sorted(dir(cl.device_info)):
                    if not name.startswith('_') and name != 'to_string':
                        try:
                            info = getattr(cl.device_info, name)
                            value = device.get_info(info)
                        except:
                            value = '<error>'
                        msg = 2 * indent + '{}: {}'
                        print(msg.format(name, value))

    def get_kernel(self, name):
        kernels = self.program.create_kernel(name)
        return CLKernel(kernels)


drv = CLDriver()


@bind_all(timings)
class CLKernel(object):
    """

    """
    def __init__(self, kernels):
        self.kernels = kernels
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

        offset = 0
        ni = inpargs[0]
        uint_t = Ctype.uint_t
        ndevs = len(drv.context.devices)

        self.local_work_sizes = []
        self.global_work_sizes = []
        self.global_work_offsets = []
        for i, kernel in enumerate(self.kernels):
            vw = kernel.vw
            lsize = kernel.lsize
            min_ngroups = kernel.min_ngroups

            n = (ni + vw - 1) // vw
            n_per_dev = (n + ndevs - 1) // ndevs

            ngroups = (n_per_dev + lsize - 1) // lsize
            ngroups = max(ngroups, min_ngroups)
            gsize = lsize * ngroups

            self.local_work_sizes.append((lsize, 1, 1))
            self.global_work_sizes.append((gsize, 1, 1))
            self.global_work_offsets.append((offset, 0, 0))

            offset += gsize

            kernel.set_arg(0, uint_t(min(offset, n)))
            for (j, buf) in enumerate(self.bufs[1:], start=1):
                kernel.set_arg(j, buf)

    def map_buffers(self):
        for (key, arg) in self.oarg.items():
            buf = self.obuf[key]
            drv.queue.enqueue_read_buffer(buf, arg, nqueues=1)
        drv.queue.wait_for_events()
        return list(self.oarg.values())

    def run(self):
        drv.queue.enqueue_nd_range_kernel(
            self.kernels,
            self.global_work_sizes,
            self.local_work_sizes,
            self.global_work_offsets)
        drv.queue.wait_for_events()


# -- End of File --
