#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
import time

from pynbody.lib.decorators import timeit
from pynbody.lib import kernels


fp_type = kernels.fp_type
kernel = kernels.p2p_pot_kernel


@timeit
def perform_calc(bi, bj):
    """Performs the calculation of the potential on a CL device."""    # TODO

    ni = len(bi)
    nj = len(bj)

    local_size = kernels.BLOCK_SIZE
    global_size = (ni + kernels.IUNROLL * local_size[0] - 1)
    global_size //= kernels.IUNROLL * local_size[0]
    global_size *= local_size[0]
    global_size = (global_size, 1, 1)

    print('lengths: ', (ni, nj))
    print('unroll: ', kernels.IUNROLL)
    print('local_size: ', local_size)
    print('global_size: ', global_size)
    print('diff: ', kernels.IUNROLL * global_size[0] - ni)

    t0 = time.time()

    iposeps2 = np.vstack((bi.pos.T, bi.eps2)).T.astype(fp_type).copy()
    jposeps2 = np.vstack((bj.pos.T, bj.eps2)).T.astype(fp_type).copy()
    jmass = bj.mass.astype(fp_type).copy()

    elapsed = time.time() - t0
    print('-'*25)
    print('Total to numpy time: {0:g} s'.format(elapsed))

    inputargs = [np.uint32(ni), np.uint32(nj), iposeps2, jposeps2, jmass]
    destshape = (ni,)
    destdtype = np.dtype(fp_type)

    ipot = kernels.exec_cl_kernel(kernel, global_size, local_size,
                                  inputargs, destshape, destdtype)

    elapsed = time.time() - t0
    print('Total execution time: {0:g} s'.format(elapsed))
    gflops = (kernel['flops']*ni*nj*1.0e-9)/elapsed
    print('Effetive Gflops/s: {0:g}'.format(gflops))

    return ipot


########## end of file ##########
