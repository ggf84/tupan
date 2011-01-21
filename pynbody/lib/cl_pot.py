#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import (print_function, with_statement)
import pyopencl as cl
import numpy as np
import time

from pynbody.lib.decorators import timeit
from pynbody.lib import kernels


ctx = kernels.ctx
queue = kernels.queue
kernel = kernels.p2p_pot_kernel['kernel']
kernel_flops = kernels.p2p_pot_kernel['flops']


@timeit
def perform_kernel_calc(iposeps2, jposeps2, jmass):
    """  """    # TODO

    ilen = len(iposeps2)
    jlen = len(jposeps2)

    unroll = kernels.IUNROLL
    local_size = kernels.BLOCK_SIZE

    global_size = (ilen + unroll * local_size[0] - 1)
    global_size //= unroll * local_size[0]
    global_size *= local_size[0]

    global_size = (global_size, 1, 1)

    print('lengths: ', (ilen, jlen))
    print('unroll: ', unroll)
    print('local_size: ', local_size)
    print('global_size: ', global_size)
    print('diff: ', unroll * global_size[0] - ilen)

    mf = cl.mem_flags
    dev_iposeps2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=iposeps2)
    dev_jposeps2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=jposeps2)
    dev_jmass = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                          hostbuf=jmass)
    dev_ipot = cl.Buffer(ctx, mf.WRITE_ONLY, size=ilen*iposeps2.dtype.itemsize)

    local_mem_size = (local_size[0] * local_size[1] * local_size[2])
    local_mem_size *= jposeps2.dtype.itemsize
    exec_evt = kernel(queue, global_size, local_size,
                      dev_iposeps2, dev_jposeps2, dev_jmass, dev_ipot,
                      np.uint32(ilen), np.uint32(jlen),
                      cl.LocalMemory(4*local_mem_size),
                      cl.LocalMemory(local_mem_size))
    exec_evt.wait()
    prog_elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
    print('-'*25)
    print('Execution time of kernel: {0:g} s'.format(prog_elapsed))

    ipot = np.empty(ilen, dtype=iposeps2.dtype)

    exec_evt = cl.enqueue_read_buffer(queue, dev_ipot, ipot)
    exec_evt.wait()
    read_elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
    print('Execution time of read_buffer: {0:g} s'.format(read_elapsed))

    gflops = (kernel_flops*ilen*jlen*1.0e-9)/prog_elapsed
    print('kernel Gflops/s: {0:g}'.format(gflops))

    return ipot


@timeit
def perform_calc(bi, bj):
    """Performs the calculation of the potential on a CL device."""    # TODO
    if kernels.ENABLE_DOUBLE_PRECISION:
        fp_type = np.float64
    else:
        fp_type = np.float32

    t0 = time.time()

    iposeps2 = np.vstack((bi.pos.T, bi.eps2)).T.astype(fp_type).copy()
    jposeps2 = np.vstack((bj.pos.T, bj.eps2)).T.astype(fp_type).copy()
    jmass = bj.mass.astype(fp_type).copy()

    elapsed = time.time() - t0
    print('-'*25)
    print('Total to numpy time: {0:g} s'.format(elapsed))

    ipot = perform_kernel_calc(iposeps2, jposeps2, jmass)

    elapsed = time.time() - t0
    print('Total execution time: {0:g} s'.format(elapsed))
    gflops = (kernel_flops*len(bi)*len(bj)*1.0e-9)/elapsed
    print('Effetive Gflops/s: {0:g}'.format(gflops))

    return ipot


########## end of file ##########
