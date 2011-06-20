#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from ggf84decor import selftimer
import numpy as _np

try:
    from pynbody.lib.clkernels import (cl_p2p_acc, cl_p2p_phi)
    cl_p2p_acc.build_kernel()
    cl_p2p_phi.build_kernel()
    _HAVE_CL = True
#    raise
except Exception as _e:
    print(_e)
    _ans = raw_input("A problem occurred with the loading of OpenCL kernels."
                     "\nAttempting to continue with C extensions on the CPU "
                     "only.\nDo you want to continue ([y]/n)? ")
    if _ans == 'n' or _ans == 'N' or _ans == 'no' or _ans == 'NO':
        import sys
        print('exiting...')
        sys.exit(0)
    else:
        _HAVE_CL = False


__all__ = ['set_acc', 'set_phi']


def _cl_set_acc(bi, bj):
    acc = cl_p2p_acc.run(bi, bj)
#    cl_p2p_acc.print_profile(len(bi), len(bj))
    return acc

def _cl_set_phi(bi, bj):
    phi = cl_p2p_phi.run(bi, bj)
#    cl_p2p_phi.print_profile(len(bi), len(bj))
    return phi


def _c_set_phi(bi, bj):
    from ._gravnewton import set_phi as c_set_phi
    phi = _np.empty(len(bi), dtype='f8')
    for i in range(len(bi)):
        phi[i] = c_set_phi(bi[i:i+1], bj)
    return phi


def _c_set_acc(bi, bj):
    from ._gravnewton import set_acc as c_set_acc
    acc = _np.empty((len(bi),4), dtype='f8')
    for i in range(len(bi)):
        acc[i,:] = c_set_acc(bi[i:i+1], bj)
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
