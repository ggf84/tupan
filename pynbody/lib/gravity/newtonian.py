#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


try:
    from pynbody.lib.kernels import clkernel as _clkernel
    _HAVE_CL = True
#    raise
except Exception as _e:
    from pynbody.lib.gravity import _gravnewton
    _HAVE_CL = False
    print(_e)
    print('Doing calculations without OpenCL...')


def _cl_set_acc(bi, bj):
    acc = _clkernel.p2p_acc.run(bi, bj)
#    _clkernel.p2p_acc.print_profile(len(bi), len(bj))
    return acc

def _cl_set_phi(bi, bj):
    phi = _clkernel.p2p_pot.run(bi, bj)
#    _clkernel.p2p_pot.print_profile(len(bi), len(bj))
    return phi


def _c_set_phi(bi, bj):
    import numpy as np
    phi = np.empty(len(bi), dtype='f8')
    for (i, obj) in enumerate(bi):
        phi[i] = _gravnewton.set_phi(obj['index'].item(),
                                     obj['mass'].item(),
                                     obj['eps2'].item(),
                                     obj['pos'].copy(),
                                     obj['vel'].copy(),
                                     bj.index.copy(),
                                     bj.mass.copy(),
                                     bj.eps2.copy(),
                                     bj.pos.copy(),
                                     bj.vel.copy())
    return phi


def _c_set_acc(bi, bj):
    import numpy as np
    acc = np.empty((len(bi),4), dtype='f8')
    for (i, obj) in enumerate(bi):
        acc[i,:] = _gravnewton.set_acc(obj['index'].item(),
                                       obj['mass'].item(),
                                       obj['eps2'].item(),
                                       obj['pos'].copy(),
                                       obj['vel'].copy(),
                                       bj.index.copy(),
                                       bj.mass.copy(),
                                       bj.eps2.copy(),
                                       bj.pos.copy(),
                                       bj.vel.copy())
    return acc


def set_phi(bi, bj):
    if _HAVE_CL:
        return _cl_set_phi(bi, bj)
    else:
        return _c_set_phi(bi, bj)


def set_acc(bi, bj):
    if _HAVE_CL:
        return _cl_set_acc(bi, bj)
    else:
        return _c_set_acc(bi, bj)


########## end of file ##########
