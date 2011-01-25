#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import os
import time
import numpy as np
import pyopencl as cl

from pynbody.lib.decorators import (selftimer, add_method_to)

path = os.path.dirname(__file__)


IUNROLL = 5                 # unroll for i-particles
JUNROLL = 16                # unroll for j-particles
BLOCK_SIZE = (256, 1, 1)
ENABLE_FAST_MATH = True
ENABLE_DOUBLE_PRECISION = True


fp_type = np.float32
if ENABLE_DOUBLE_PRECISION:
    fp_type = np.float64

dtype = np.dtype(fp_type)


ctx = cl.create_some_context()
_properties = cl.command_queue_properties.PROFILING_ENABLE
queue = cl.CommandQueue(ctx, properties=_properties)


@selftimer()
class Kernels(object):
    """
    This class serves as an abstraction layer to manage CL kernels.
    """

    def __init__(self, fname, options=' '):
        """
        Setup a kernel from .cl source file.
        """
        def get_from(source, pattern):
            for line in source.splitlines():
                if pattern in line:
                    return line.split()[-1]
        self.name = fname.split('.')[0]
        with open(path+'/'+fname, 'r') as f:
            self.source = f.read()
            self.flops = int(get_from(self.source, 'Total flop count'))
            self.output_shape = get_from(self.source, 'Output shape')
            prog = cl.Program(ctx, self.source)
            if ENABLE_DOUBLE_PRECISION:
                options += ' -D DOUBLE'
            if ENABLE_FAST_MATH:
                options += ' -cl-fast-relaxed-math'
            prog.build(options=options)
            self.kernel = cl.Kernel(prog, self.name)

    @selftimer()
    def call_kernel(self, global_size, local_size, inputargs, destshape,
                    local_mem_size, gflops_count):
        """
        Call a kernel on a CL device.
        """
        mf = cl.mem_flags
        dev_args = [global_size, local_size]
        for item in inputargs:
            if isinstance(item, np.ndarray):
                dev_args.append(cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=item))
            else:
                dev_args.append(item)

        destsize = reduce(lambda x, y: x * y, destshape) * dtype.itemsize
        dev_dest = cl.Buffer(ctx, mf.WRITE_ONLY, size=destsize)
        dev_args.append(dev_dest)

        for size in local_mem_size:
            dev_args.append(cl.LocalMemory(size))

        exec_evt = self.kernel(queue, *dev_args)

        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
        print('Execution time of CL kernel: {0:g} s'.format(elapsed))
        print('CL kernel Gflops/s: {0:g}'.format(gflops_count/elapsed))

        dest = np.empty(destshape, dtype=dtype)

        cl.enqueue_read_buffer(queue, dev_dest, dest).wait()

        return dest


    def run(self, bi, bj):
        """
        Runs the calculation on a CL device.
        """

        ni = len(bi)
        nj = len(bj)

        local_size = BLOCK_SIZE
        global_size = (ni + IUNROLL * local_size[0] - 1)
        global_size //= IUNROLL * local_size[0]
        global_size *= local_size[0]
        global_size = (global_size, 1, 1)

        print('lengths: ', (ni, nj))
        print('unroll: ', IUNROLL)
        print('local_size: ', local_size)
        print('global_size: ', global_size)
        print('diff: ', IUNROLL * global_size[0] - ni)

        t0 = time.time()

        iposeps2 = np.vstack((bi.pos.T, bi.eps2)).T.copy().astype(fp_type)
        jposeps2 = np.vstack((bj.pos.T, bj.eps2)).T.copy().astype(fp_type)
        jmass = bj.mass.copy().astype(fp_type)

        elapsed = time.time() - t0
        print('-'*25)
        print('Total to numpy time: {0:g} s'.format(elapsed))

        inputargs = (np.uint32(ni), np.uint32(nj), iposeps2, jposeps2, jmass)
        destshape = eval(self.output_shape.format(ni=ni))
        mem_size = reduce(lambda x, y: x * y, local_size) * dtype.itemsize
        local_mem_size = (4*mem_size, mem_size)
        gflops_count = self.flops * ni * nj * 1.0e-9

        dest = self.call_kernel(*[global_size, local_size, inputargs,
                                  destshape, local_mem_size, gflops_count])

        elapsed = self.call_kernel.func_closure[0].cell_contents.elapsed
        print('Total call_kernel time: {0:g} s'.format(elapsed))
        print('call_kernel Gflops/s: {0:g}'.format(gflops_count/elapsed))

        elapsed = time.time() - t0
        print('Total execution time: {0:g} s'.format(elapsed))
        print('Effetive Gflops/s: {0:g}'.format(gflops_count/elapsed))

        return dest






options = '-D IUNROLL={iunroll} -D JUNROLL={junroll}'.format(iunroll=IUNROLL,
                                                             junroll=JUNROLL)

# setup the potential kernel
p2p_pot_kernel = Kernels('p2p_pot_kernel.cl', options)

# setup the acceleration kernel
p2p_acc_kernel = Kernels('p2p_acc_kernel.cl', options)

# setup the acceleration kernel (gpugems3)
p2p_acc_kernel_gpugems3 = Kernels('p2p_acc_kernel_gpugems3.cl', options)



@add_method_to(p2p_acc_kernel_gpugems3)
def print_name(self):
    """doc of print_name"""
    print('*** name: ', self.name, self.flops, '***')
p2p_acc_kernel_gpugems3.print_name()



########## end of file ##########
