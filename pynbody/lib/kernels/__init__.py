#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


#from __future__ import (print_function, with_statement)
import os
import pyopencl as cl


from pynbody.lib.decorators import timeit

path = os.path.dirname(__file__)


IUNROLL = 5                 # unroll for i-particles
JUNROLL = 16                # unroll for j-particles
BLOCK_SIZE = (256, 1, 1)
ENABLE_FAST_MATH = True
ENABLE_DOUBLE_PRECISION = True


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


# setup the potential kernel
p2p_pot_kernel = setup_cl_kernel('p2p_pot_kernel.cl', 15)

# setup the acceleration kernel
p2p_acc_kernel = setup_cl_kernel('p2p_acc_kernel.cl', 23)

# setup the acceleration kernel (gpugems3)
p2p_acc_kernel_gpugems3 = setup_cl_kernel('p2p_acc_kernel_gpugems3.cl', 23)


########## end of file ##########
