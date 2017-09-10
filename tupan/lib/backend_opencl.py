# -*- coding: utf-8 -*-
#

"""
This module implements the OpenCL backend to call CL-extensions.
"""

import os
import logging
import numpy as np
import pyopencl as cl
from ..config import cli, Ctype


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'src')

USE_HOST_FLAG = cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR
COPY_HOST_FLAG = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
ALLOC_HOST_FLAG = cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR

MAP_FLAGS = cl.map_flags.READ | cl.map_flags.WRITE


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
        self.default_queue = Queue(self.cl_context)
        self.devices = [Device(self.cl_context, cl_device)
                        for cl_device in cl_devices]

    def to_buf(self, array):
        return cl.Buffer(self.cl_context, COPY_HOST_FLAG, hostbuf=array)


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

    def build(self, fp=cli.fp):
        dev = self.cl_device

        simd = (dev.preferred_vector_width_float
                if fp == 32
                else dev.preferred_vector_width_double)

        wpt = 1                         # work per thread
        nwarps = 1                      # number of warps in a work group
        nlanes = 1                      # number of lanes in a warp/wavefront
        wgsize = 1                      # work_group_size
        ngroup = dev.max_compute_units  # num_groups

        if dev.type == cl.device_type.CPU:
            wpt *= 8
            simd *= 1
            nwarps *= 4
            nlanes *= 1
            wgsize *= nwarps * nlanes
            ngroup *= 4

        if dev.type == cl.device_type.GPU:
            wpt *= 2
            simd *= 2 if fp == 32 else 1
            nwarps *= 2
            nlanes *= 32
            wgsize *= nwarps * nlanes
            ngroup *= 128

        # setting program options
        options = ' -D WPT={}'.format(wpt)
        options += ' -D SIMD={}'.format(simd)
        options += ' -D NLANES={}'.format(nlanes)
        options += ' -D WGSIZE={}'.format(wgsize)
        options += ' -D CONFIG_USE_OPENCL'
        if fp == 64:
            options += ' -D CONFIG_USE_DOUBLE'
        options += ' -I {path}'.format(path=PATH)
        options += ' -cl-fast-relaxed-math'
#        options += ' -cl-opt-disable'

        self.cl_program.build(options=options, devices=[dev])

        kernels = self.cl_program.all_kernels()
        for kernel in kernels:
            name = kernel.function_name
            kernel.wgsize = wgsize
            kernel.ngroup = ngroup
            kernel.name = name
#            kwgi = cl.kernel_work_group_info
#            LOGGER.debug(
#                "\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s",
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


class CLDriver(object):
    """

    """
    def __init__(self):
        self.platform = Platform()
        self.context = self.platform.context
        self.queue = self.context.default_queue

        self.to_int = Ctype.int_t
        self.to_uint = Ctype.uint_t
        self.to_real = Ctype.real_t
        self.to_buf = self.context.to_buf


drv = CLDriver()


class CLKernel(object):
    """

    """
    def __init__(self, name):
        triangle = rectangle = name
        if 'kepler' not in name:
            triangle += '_triangle'
            rectangle += '_rectangle'

        self.kernels_t = [device.program.kernel[triangle]
                          for device in drv.context.devices]
        self.kernels_r = [device.program.kernel[rectangle]
                          for device in drv.context.devices]

        self.queues = [device.queue
                       for device in drv.context.devices]

        self.to_int = drv.to_int
        self.to_uint = drv.to_uint
        self.to_real = drv.to_real

#   #1
    def to_ibuf(self, obj):
        obj.buf = drv.to_buf(obj.ptr)
        return obj.buf

    to_obuf = to_ibuf

    def map_buf(self, obj,
                default_queue=drv.context.default_queue):
        default_queue.enqueue_read_buffer(obj.buf, obj.ptr)
        default_queue.wait_for_events()

#   #2
#    def to_ibuf(self, obj,
#                default_queue=drv.context.default_queue):
#        default_queue.enqueue_read_buffer(obj.ptr, obj.buf)
#        default_queue.wait_for_events()
#        return obj.buf
#
#    to_obuf = to_ibuf
#
#    def map_buf(self, obj,
#                default_queue=drv.context.default_queue):
#        default_queue.enqueue_read_buffer(obj.buf, obj.ptr)
#        default_queue.wait_for_events()

#   #3
#    def to_ibuf(self, obj):
#        return obj.buf
#
#    to_obuf = to_ibuf
#
#    def map_buf(self, obj):
#        pass


#   #4 or #5
#    def to_ibuf(self, obj):
#        return obj.buf
#
#    def to_obuf(self, obj):
#        del obj.ptr
#        return obj.buf
#
#    def map_buf(self, obj,
#                ctx=drv.context.cl_context,
#                queue=drv.queue.cl_queue):
#        obj.ptr, ev = cl.enqueue_map_buffer(queue, obj.buf,
#                                            flags=MAP_FLAGS,
#                                            offset=0,
#                                            shape=obj.shape,
#                                            dtype=obj.dtype,
#                                            is_blocking=False)
#        ev.wait()

# --

    def triangle(self, *args):
        for queue, kernel in zip(self.queues, self.kernels_t):
            wgsize = kernel.wgsize
            ngroup = kernel.ngroup

            gwsize = ngroup * wgsize

            local_work_size = (wgsize, 1, 1)
            global_work_size = (gwsize, 1, 1)

            for i, arg in enumerate(args):
                kernel.set_arg(i, arg)

            queue.enqueue_nd_range_kernel(
                kernel,
                global_work_size,
                local_work_size,
            )

        drv.context.default_queue.wait_for_events()

    def rectangle(self, *args):
        for queue, kernel in zip(self.queues, self.kernels_r):
            wgsize = kernel.wgsize
            ngroup = kernel.ngroup

            gwsize = ngroup * wgsize

            local_work_size = (wgsize, 1, 1)
            global_work_size = (gwsize, 1, 1)

            for i, arg in enumerate(args):
                kernel.set_arg(i, arg)

            queue.enqueue_nd_range_kernel(
                kernel,
                global_work_size,
                local_work_size,
            )

        drv.context.default_queue.wait_for_events()


def to_clbuf(ary,
             ctx=drv.context.cl_context,
             queue=drv.queue.cl_queue):
    nbytes = ary.nbytes
    if not nbytes:
        return ary, None

#   #1
    ptr = np.array(ary, copy=False, order='C')
    buf = None
    return ptr, buf

#   #2
#    ptr = np.array(ary, copy=False, order='C')
#    buf = cl.Buffer(ctx, ALLOC_HOST_FLAG, size=nbytes)
#    return ptr, buf

#   #3
#    ptr = np.array(ary, copy=False, order='C')
#    buf = cl.Buffer(ctx, USE_HOST_FLAG, hostbuf=ptr)
#    return ptr, buf

#   #4
#    ptr = np.array(ary, copy=False, order='C')
#    buf = cl.Buffer(ctx, COPY_HOST_FLAG, hostbuf=ptr)
#    ptr, ev = cl.enqueue_map_buffer(queue, buf,
#                                    flags=MAP_FLAGS,
#                                    offset=0,
#                                    shape=ary.shape,
#                                    dtype=ary.dtype,
#                                    is_blocking=False)
#    ev.wait()
#    return ptr, buf

#   #5
#    buf = cl.Buffer(ctx, ALLOC_HOST_FLAG, size=nbytes)
#    ptr, ev = cl.enqueue_map_buffer(queue, buf,
#                                    flags=MAP_FLAGS,
#                                    offset=0,
#                                    shape=ary.shape,
#                                    dtype=ary.dtype,
#                                    is_blocking=False)
#    ev.wait()
#    ptr[...] = ary
#    return ptr, buf


# -- End of File --
