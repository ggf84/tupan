# -*- coding: utf-8 -*-
#

"""
This module implements the OpenCL backend to call CL-extensions.
"""

import os
import logging
import pyopencl as cl
from ..config import cli, Ctype


LOGGER = logging.getLogger(__name__)

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'src')

RW_FLAG = cl.mem_flags.READ_WRITE
USE_HOST_FLAG = RW_FLAG | cl.mem_flags.USE_HOST_PTR
COPY_HOST_FLAG = RW_FLAG | cl.mem_flags.COPY_HOST_PTR
ALLOC_HOST_FLAG = RW_FLAG | cl.mem_flags.ALLOC_HOST_PTR

#MAP_FLAGS = cl.map_flags.READ
#MAP_FLAGS = cl.map_flags.WRITE
MAP_FLAGS = cl.map_flags.READ | cl.map_flags.WRITE


def get_platform(idx=0):
    platforms = cl.get_platforms()
    if len(platforms) > 1:
        print("Choose platform:")
        for i, platform in enumerate(platforms):
            print(str([i]) + ' ' + str(platform))
        try:
            idx = int(input('Choice [0]: '))
        except:
            pass
    return platforms[idx]


def get_context(platform):
    devs = platform.get_devices()

#    # emulate multiple devices
#    dev = devs[0]
#    devs = dev.create_sub_devices(
#        [cl.device_partition_property.EQUALLY, 1]
#    )
#    devs = dev.create_sub_devices(
#        [cl.device_partition_property.BY_COUNTS, 1, 1]
#    )

    return cl.Context(devices=devs)


def get_queues(ctx):
    cqp = cl.command_queue_properties
    props = 0
    props |= cqp.OUT_OF_ORDER_EXEC_MODE_ENABLE
#    props |= cqp.PROFILING_ENABLE

    return [cl.CommandQueue(ctx, device=dev, properties=props)
            for dev in ctx.devices]


def get_program(ctx):
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

    return cl.Program(ctx, source)


def build(prog, dev, fp=cli.fp):
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
#    options += ' -cl-opt-disable'

    prog.build(options=options, devices=[dev])

    kernels = {}
    for kernel in prog.all_kernels():
        name = kernel.function_name
        kernels[name] = (kernel, wgsize, ngroup)
#        kwgi = cl.kernel_work_group_info
#        LOGGER.debug(
#            "\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s",
#            name,
#            kernel.get_work_group_info(
#                kwgi.COMPILE_WORK_GROUP_SIZE,
#                dev),
#            kernel.get_work_group_info(
#                kwgi.LOCAL_MEM_SIZE,
#                dev),
#            kernel.get_work_group_info(
#                kwgi.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
#                dev),
#            kernel.get_work_group_info(
#                kwgi.PRIVATE_MEM_SIZE,
#                dev),
#            kernel.get_work_group_info(
#                kwgi.WORK_GROUP_SIZE,
#                dev),
#            dev
#        )

#    for name, (kernel, wgsize, ngroup) in kernels.items():
#        print(kernel, kernel.attributes, name)
#    print('-'*48)

    return kernels


def get_kernels(prog, queues):
    return [build(prog, queue.device) for queue in queues]


class CLDriver(object):
    """

    """
    def __init__(self):
        self.plat = get_platform()
        self.ctx = get_context(self.plat)
        self.prog = get_program(self.ctx)
        self.queues = get_queues(self.ctx)
        self.kernels = get_kernels(self.prog, self.queues)

        self.to_int = Ctype.int_t
        self.to_uint = Ctype.uint_t
        self.to_real = Ctype.real_t


drv = CLDriver()


class CLKernel(object):
    """

    """
    def __init__(self, name):
        triangle = rectangle = name
        if 'kepler' not in name:
            triangle += '_triangle'
            rectangle += '_rectangle'

        self.kernels = {
            'triangle': [kernel[triangle] for kernel in drv.kernels],
            'rectangle': [kernel[rectangle] for kernel in drv.kernels],
        }

        self.to_int = drv.to_int
        self.to_uint = drv.to_uint
        self.to_real = drv.to_real

    def sync(self):
        for queue in drv.queues:
            queue.finish()

    def run(self, *args, name=None):
        kernels = self.kernels[name]
        for queue, (kernel, wgsize, ngroup) in zip(drv.queues, kernels):
            gwsize = ngroup * wgsize

            local_work_size = (wgsize, 1, 1)
            global_work_size = (gwsize, 1, 1)

            for i, arg in enumerate(args):
                kernel.set_arg(i, arg)

            cl.enqueue_nd_range_kernel(
                queue,
                kernel,
                global_work_size,
                local_work_size,
            )
            queue.flush()

        self.sync()

    def triangle(self, *args):
        self.run(*args, name='triangle')

    def rectangle(self, *args):
        self.run(*args, name='rectangle')

#   #1
    def to_buf(self, obj):
        if obj.buf is None:
            obj.buf = cl.Buffer(drv.ctx, COPY_HOST_FLAG, hostbuf=obj.ptr)
        obj.ptr = None
        return obj.buf

    def map_buf(self, obj):
        if obj.ptr is None:
            ptr, ev = cl.enqueue_map_buffer(drv.queues[0],
                                            obj.buf,
                                            flags=MAP_FLAGS,
                                            offset=0,
                                            shape=obj.shape,
                                            dtype=obj.dtype,
                                            is_blocking=False)
            obj.ptr = ptr


#   #2
#    def to_buf(self, obj,
#                queue=drv.queues[0]):
#        cl.enqueue_copy(queue, obj.buf, obj.ptr,
#                        is_blocking=False)
#        return obj.buf
#
#    def map_buf(self, obj,
#                queue=drv.queues[0]):
#        cl.enqueue_copy(queue, obj.ptr, obj.buf,
#                        is_blocking=False)


#   #3
#    def to_buf(self, obj):
#        return obj.buf
#
#    def map_buf(self, obj):
#        pass


#   #4 or #5
#    def to_buf(self, obj):
#        obj.ptr = None
#        return obj.buf
#
#    def map_buf(self, obj,
#                queue=drv.queues[0]):
#        ptr, ev = cl.enqueue_map_buffer(queue, obj.buf,
#                                        flags=MAP_FLAGS,
#                                        offset=0,
#                                        shape=obj.shape,
#                                        dtype=obj.dtype,
#                                        is_blocking=False)
#        obj.ptr = ptr


#   #6
#    def to_buf(self, obj,
#                queue=drv.queues[0]):
#        cl.enqueue_copy(queue, obj.buf[1], obj.ptr,
#                        is_blocking=False)
#        return obj.buf[1]
#
#    def map_buf(self, obj,
#                queue=drv.queues[0]):
#        cl.enqueue_copy(queue, obj.ptr, obj.buf[1],
#                        is_blocking=False)


def to_clbuf(ary,
             ctx=drv.ctx,
             queue=drv.queues[0]):
#   #1
    ptr = ary
    buf = None
    return ptr, buf

#    nbytes = ary.nbytes
#    if not nbytes:
#        return ary, None
#
#   #2
#    ptr = ary
#    buf = cl.Buffer(ctx, RW_FLAG, size=nbytes)
#    return ptr, buf

#   #3
#    ptr = ary
#    buf = cl.Buffer(ctx, USE_HOST_FLAG, hostbuf=ptr)
#    return ptr, buf

#   #4
#    ptr = ary
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

#   #6
#    dbuf = cl.Buffer(ctx, RW_FLAG, size=nbytes)
#    hbuf = cl.Buffer(ctx, ALLOC_HOST_FLAG, size=nbytes)
#    ptr, ev = cl.enqueue_map_buffer(queue, hbuf,
#                                    flags=MAP_FLAGS,
#                                    offset=0,
#                                    shape=ary.shape,
#                                    dtype=ary.dtype,
#                                    is_blocking=False)
#    ev.wait()
#    ptr[...] = ary
#    return ptr, (hbuf, dbuf)


# -- End of File --
