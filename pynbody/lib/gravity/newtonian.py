#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from ggf84decor import selftimer
import numpy as _np

from pynbody.lib.clkernels import (cl_p2p_acc, cl_p2p_phi)


__all__ = ['set_acc', 'set_phi']


def cl_set_acc(bi, bj):
    acc = cl_p2p_acc.run(bi, bj)
#    cl_p2p_acc.print_profile(len(bi), len(bj))
    return acc

def cl_set_phi(bi, bj):
    phi = cl_p2p_phi.run(bi, bj)
#    cl_p2p_phi.print_profile(len(bi), len(bj))
    return phi


def c_set_phi(bi, bj):
    from _gravnewton import set_phi as c_p2p_phi
    phi = _np.empty(len(bi), dtype='f8')
    for i in range(len(bi)):
        phi[i] = c_p2p_phi(bi[i:i+1], bj)
    return phi


def c_set_acc(bi, bj):
    from _gravnewton import set_acc as c_p2p_acc
    acc = _np.empty((len(bi),4), dtype='f8')
    for i in range(len(bi)):
        acc[i,:] = c_p2p_acc(bi[i:i+1], bj)
    return acc


@selftimer
def set_phi(bi, bj):
    if cl_p2p_phi.is_available:
        return cl_set_phi(bi, bj)
    else:
        return c_set_phi(bi, bj)


@selftimer
def set_acc(bi, bj):
    if cl_p2p_acc.is_available:
        return cl_set_acc(bi, bj)
    else:
        return c_set_acc(bi, bj)


########## end of file ##########
