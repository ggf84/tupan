#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from ggf84decor import selftimer
import numpy as _np

try:
    import pynbody.lib.clkernels as clkernels
    _HAVE_CL = True
#    raise
except Exception as _e:
    import _gravnewton
    print(_e)
    _ans = raw_input("A problem occurred with the loading of OpenCL kernels."
                     "\nAttempting to continue with C extensions on the CPU "
                     "only.\nDo you want to continue [y/n]? ")
    if _ans == 'y' or _ans == 'Y' or _ans == 'yes' or _ans == 'YES':
        _HAVE_CL = False
    else:
        import sys
        print('exiting...')
        sys.exit(0)


def _cl_set_acc(bi, bj):
    acc = clkernels.p2p_acc.run(bi, bj)
#    clkernels.p2p_acc.print_profile(len(bi), len(bj))
    return acc

def _cl_set_phi(bi, bj):
    phi = clkernels.p2p_phi.run(bi, bj)
#    clkernels.p2p_phi.print_profile(len(bi), len(bj))
    return phi


def _c_set_phi(bi, bj):
    phi = _np.empty(len(bi), dtype='f8')
    for i in range(len(bi)):
        phi[i] = _gravnewton.set_phi(bi[i:i+1], bj)
    return phi


def _c_set_acc(bi, bj):
    acc = _np.empty((len(bi),4), dtype='f8')
    for i in range(len(bi)):
        acc[i,:] = _gravnewton.set_acc(bi[i:i+1], bj)
    return acc


@selftimer
def set_phi(bi, bj):
    if _HAVE_CL:
        return _cl_set_phi(bi, bj)
    else:
        return _c_set_phi(bi, bj)


@selftimer
def set_acc(bi, bj):
    if _HAVE_CL:
        return _cl_set_acc(bi, bj)
    else:
        return _c_set_acc(bi, bj)


########## end of file ##########
