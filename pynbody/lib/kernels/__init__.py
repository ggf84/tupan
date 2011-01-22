#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import os
import numpy as np
import pyopencl as cl

from pynbody.lib.decorators import timeit

path = os.path.dirname(__file__)


IUNROLL = 5                 # unroll for i-particles
JUNROLL = 16                # unroll for j-particles
BLOCK_SIZE = (256, 1, 1)
ENABLE_FAST_MATH = True
ENABLE_DOUBLE_PRECISION = True


fp_type = np.float32
if ENABLE_DOUBLE_PRECISION:
    fp_type = np.float64


ctx = cl.create_some_context()
_properties = cl.command_queue_properties.PROFILING_ENABLE
queue = cl.CommandQueue(ctx, properties=_properties)


@timeit
def setup_cl_kernel(fname, flops):
    """  """    # TODO

    kprops = {'flops': flops, 'name': fname.split('.')[0]}

    with open(path+'/'+fname, 'r') as f:
        kprops['source'] = f.read()
        prog = cl.Program(ctx, kprops['source'])
        options = '-D UNROLL_SIZE_I={iunroll} \
                   -D UNROLL_SIZE_J={junroll}'.format(iunroll=IUNROLL,
                                                      junroll=JUNROLL)
        if ENABLE_DOUBLE_PRECISION:
            options += ' -D DOUBLE'
        if ENABLE_FAST_MATH:
            options += ' -cl-fast-relaxed-math'
        prog.build(options=options)
        kprops['kernel'] = cl.Kernel(prog, kprops['name'])

    return kprops


@timeit
def exec_cl_kernel(kernel, global_size, local_size,
                   inputargs, destshape, destdtype):
    """  """    # TODO

    mf = cl.mem_flags
    dev_args = [global_size, local_size]
    for item in inputargs:
        if isinstance(item, np.ndarray):
            dev_args.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=item))
        else:
            dev_args.append(item)

    destsize=reduce(lambda x, y: x*y, destshape)*destdtype.itemsize
    dev_dest = cl.Buffer(ctx, mf.WRITE_ONLY, size=destsize)
    dev_args.append(dev_dest)

    local_mem_size = reduce(lambda x, y: x*y, local_size)
    local_mem_size *= np.dtype(fp_type).itemsize
    dev_args.append(cl.LocalMemory(4*local_mem_size))
    dev_args.append(cl.LocalMemory(local_mem_size))

    exec_evt = kernel['kernel'](queue, *dev_args)

    exec_evt.wait()
    prog_elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
    print('-'*25)
    print('Execution time of kernel: {0:g} s'.format(prog_elapsed))

    dest = np.empty(destshape, dtype=destdtype)

    exec_evt = cl.enqueue_read_buffer(queue, dev_dest, dest)
    exec_evt.wait()
    read_elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
    print('Execution time of read_buffer: {0:g} s'.format(read_elapsed))

    gflops = (kernel['flops']*inputargs[0]*inputargs[1]*1.0e-9)/prog_elapsed
    print('kernel Gflops/s: {0:g}'.format(gflops))

    return dest


# setup the potential kernel
p2p_pot_kernel = setup_cl_kernel('p2p_pot_kernel.cl', 15)

# setup the acceleration kernel
p2p_acc_kernel = setup_cl_kernel('p2p_acc_kernel.cl', 23)

# setup the acceleration kernel (gpugems3)
p2p_acc_kernel_gpugems3 = setup_cl_kernel('p2p_acc_kernel_gpugems3.cl', 23)


########## end of file ##########
