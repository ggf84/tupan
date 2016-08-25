# -*- coding: utf-8 -*-
#

"""
This module implements the OpenCL backend to call CL-extensions.
"""

import os
import logging
import pyopencl as cl
from functools import partial
from collections import defaultdict
from ..config import options, get_cache_dir
from .utils.ctype import Ctype


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'src')


class Context(object):
    """

    """
    def __init__(self, cl_platform):
        self.alignment = 128    # Minimum alignment (bytes) for any datatype
        cl_devices = cl_platform.get_devices()

#        # emulate multiple devices
#        cl_devices = cl_devices[0].create_sub_devices(
#            [cl.device_partition_property.EQUALLY, 1])
#        cl_devices = cl_devices[0].create_sub_devices(
#            [cl.device_partition_property.BY_COUNTS, 1, 1, 1])

        self.cl_context = cl.Context(devices=cl_devices)
        self.default_queue = Queue(self.cl_context)
        self.devices = [Device(self.cl_context, cl_device)
                        for cl_device in cl_devices]

        self.ibuf = defaultdict(partial(cl.Buffer, self.cl_context,
                                        cl.mem_flags.READ_ONLY,
                                        size=1))

        self.obuf = defaultdict(partial(cl.Buffer, self.cl_context,
                                        cl.mem_flags.WRITE_ONLY,
                                        size=1))

        self.reset_buf_counts()

    def to_ibuf(self, ary):
        flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR
#        flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        buf = cl.Buffer(self.cl_context, flags, hostbuf=ary)
        self.ibuf[self.ibuf_count] = buf
        self.ibuf_count += 1
        return buf

    def to_obuf(self, ary):
        flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR
#        flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR
        buf = cl.Buffer(self.cl_context, flags, hostbuf=ary)
        self.obuf[self.obuf_count] = buf
        self.obuf_count += 1
        return buf

    def get_ibuf(self, ary):
        flags = cl.mem_flags.READ_WRITE
        align = self.alignment
        size = ((ary.nbytes + align - 1) // align) * align
        buf = self.ibuf[self.ibuf_count]
        if size > buf.size:
            buf = cl.Buffer(self.cl_context, flags, size=size)
        self.default_queue.enqueue_write_buffer(buf, ary, is_blocking=True)
        self.ibuf[self.ibuf_count] = buf
        self.ibuf_count += 1
        return buf

    def get_obuf(self, ary):
        flags = cl.mem_flags.READ_WRITE
        align = self.alignment
        size = ((ary.nbytes + align - 1) // align) * align
        buf = self.obuf[self.obuf_count]
        if size > buf.size:
            buf = cl.Buffer(self.cl_context, flags, size=size)
        self.default_queue.enqueue_write_buffer(buf, ary, is_blocking=True)
        self.obuf[self.obuf_count] = buf
        self.obuf_count += 1
        return buf

    def reset_buf_counts(self):
        self.ibuf_count = 0
        self.obuf_count = 0


class Queue(object):
    """

    """
    events = []

    def __init__(self, cl_context, cl_device=None,
                 out_of_order=False, profiling=False):
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
        if self.events:
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
                                global_work_offset=None,
                                wait_for=None):
        event = cl.enqueue_nd_range_kernel(
            self.cl_queue,
            kernel,
            global_work_size,
            local_work_size,
            global_work_offset=global_work_offset,
            wait_for=wait_for
        )
        self.cl_queue.flush()
        self.events.append(event)


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


class Program(object):
    """

    """
    def __init__(self, cl_context, cl_device):
        fnames = (
            'phi_kernel.cl',
            'acc_kernel.cl',
            'acc_jrk_kernel.cl',
            'snp_crk_kernel.cl',
            'tstep_kernel.cl',
            'pnacc_kernel.cl',
            'nreg_kernels.cl',
            'sakura_kernel.cl',
        )

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
        simd = (self.cl_device.preferred_vector_width_float
                if fpwidth == 'fp32'
                else self.cl_device.preferred_vector_width_double)
        lsize = 1
        wsize = 8
        if self.cl_device.type == cl.device_type.CPU:
            lsize *= 2
            wsize *= 2
        if self.cl_device.type == cl.device_type.GPU:
            lsize *= 256
            wsize *= 256
        fast_local_mem = True

        # setting program options
        opts = ' -D SIMD={}'.format(simd)
        opts += ' -D LSIZE={}'.format(lsize)
        opts += ' -D CONFIG_USE_OPENCL'
        if fpwidth == 'fp64':
            opts += ' -D CONFIG_USE_DOUBLE'
        if fast_local_mem:
            opts += ' -D FAST_LOCAL_MEM'
        opts += ' -I {path}'.format(path=PATH)
        opts += ' -cl-fast-relaxed-math'
#        opts += ' -cl-opt-disable'

        self.cl_program.build(options=opts,
                              devices=[self.cl_device],
                              cache_dir=get_cache_dir(fpwidth))

        kernels = self.cl_program.all_kernels()
        for kernel in kernels:
            name = kernel.function_name
            kernel.wsize = wsize
            kernel.lsize = lsize
            kernel.name = name
#            kwgi = cl.kernel_work_group_info
#            LOGGER.debug(
#                "CL '%s' info: %s %s %s %s %s on %s",
#                name,
#                kernel.get_work_group_info(
#                    kwgi.COMPILE_WORK_GROUP_SIZE,
#                    self.cl_device),
#                kernel.get_work_group_info(
#                    kwgi.LOCAL_MEM_SIZE,
#                    self.cl_device),
#                kernel.get_work_group_info(
#                    kwgi.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
#                    self.cl_device),
#                kernel.get_work_group_info(
#                    kwgi.PRIVATE_MEM_SIZE,
#                    self.cl_device),
#                kernel.get_work_group_info(
#                    kwgi.WORK_GROUP_SIZE,
#                    self.cl_device),
#                self.cl_device
#            )

        self.kernel = {kernel.name: kernel for kernel in kernels}

#        for kernel in self.kernel.values():
#            print(kernel, kernel.attributes, kernel.name)
#        print('-'*48)

        return self


class Platform(object):
    """

    """
    def __init__(self, idx=0):
        cl_platforms = cl.get_platforms()
        if len(cl_platforms) > 1:
            print("Choose platform:")
            for i, cl_platform in enumerate(cl_platforms):
                print(str([i]) + ' ' + str(cl_platform))
            try:
                idx = int(input('Choice [0]: '))
            except:
                pass
        self.cl_platform = cl_platforms[idx]
        self.context = Context(self.cl_platform)

    def get_kernel(self, name):
        return CLKernel(name)


drv = Platform()


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

    def make_struct(self, name, **kwargs):
        real_t = Ctype.real_t
        uint_t = Ctype.uint_t
        formats = {'CLIGHT': [('inv1', real_t), ('inv2', real_t),
                              ('inv3', real_t), ('inv4', real_t),
                              ('inv5', real_t), ('inv6', real_t),
                              ('inv7', real_t), ('order', uint_t)]}

        import numpy as np
        dtype = np.dtype(formats[name], align=True)
        fields = tuple(kwargs[name] for name in dtype.names)
#        return {'struct': np.array(fields, dtype=dtype)}
        return np.array(fields, dtype=dtype)

    def set_args(self, inpargs, outargs):
        bufs = []

        drv.context.reset_buf_counts()

        if self.inptypes is None:
            types = []
            for arg in inpargs:
                if isinstance(arg, int):
                    # sizeof(int_t) == sizeof(uint_t)
                    types.append(Ctype.uint_t)
                elif isinstance(arg, float):
                    types.append(Ctype.real_t)
#                elif isinstance(arg, dict):
#                    types.append(lambda x: x['struct'])
                else:
                    iptr = drv.context.to_ibuf
                    types.append(iptr)
            self.inptypes = types

        if self.outtypes is None:
            optr = drv.context.to_obuf
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
        flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR
        for (key, arg) in self.oarg.items():
            buf = self.obuf[key]
            if buf.flags == flags:
                drv.context.default_queue.enqueue_read_buffer(buf, arg)
        drv.context.default_queue.wait_for_events()
        return list(self.oarg.values())

    def run(self):
        ni = self.iarg[0]
        name = self.name
        uint_t = Ctype.uint_t
#        ndevs = len(drv.context.devices)
#        dn = (ni + ndevs - 1) // ndevs

        for device in drv.context.devices:
            kernel = device.program.kernel[name]

            wsize = kernel.wsize
            lsize = kernel.lsize

            gsize = wsize * lsize

            local_work_size = (lsize, 1, 1)
            global_work_size = (gsize, 1, 1)

            kernel.set_arg(0, uint_t(ni))
            for (j, buf) in enumerate(self.bufs[1:], start=1):
                kernel.set_arg(j, buf)

            device.queue.enqueue_nd_range_kernel(
                kernel,
                global_work_size,
                local_work_size,
            )

        drv.context.default_queue.wait_for_events()


# -- End of File --
