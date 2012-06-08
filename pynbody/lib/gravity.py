#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements a minimal class for gravitational interactions
between particles in Newtonian and post-Newtonian approach.
"""


import sys
import numpy as np
from .extensions import ext
from .utils.timing import decallmethods, timings


__all__ = ["Gravity", "gravitation"]


@decallmethods(timings)
class Clight(object):
    """
    This class holds the PN-order and some inverse powers of clight.
    """
    def __init__(self, pn_order, clight):
        self.pn_order = int(pn_order)
        self.inv1 = 1.0/float(clight)
        self.inv2 = self.inv1**2
        self.inv3 = self.inv1**3
        self.inv4 = self.inv1**4
        self.inv5 = self.inv1**5
        self.inv6 = self.inv1**6
        self.inv7 = self.inv1**7


class Phi(object):
    """

    """
    def __init__(self):
        self.kernel = ext.p2p_phi_kernel
        self.kernel.local_size = 384
        self.kernel.set_arg('LMEM', 5, 8)

    def set_arg(self, mode, i, arg):
        self.kernel.set_arg(mode, i, arg)

    @property
    def global_size(self):
        return self.kernel.global_size

    @global_size.setter
    def global_size(self, value):
        self.kernel.global_size = value

    def run(self):
        self.kernel.run()
        return self

    def get_result(self):
        result = self.kernel.get_result()[0]
        return result


class Acc(object):
    """

    """
    def __init__(self):
        self.kernel = ext.p2p_acc_kernel
        self.kernel.local_size = 384
        self.kernel.set_arg('LMEM', 5, 8)

    def set_arg(self, mode, i, arg):
        self.kernel.set_arg(mode, i, arg)

    @property
    def global_size(self, value):
        self.kernel.global_size = value

    def run(self, ni):
        self.kernel.global_size = ni
        self.kernel.run()
        return self

    def get_result(self):
        result = self.kernel.get_result()[0]
        return result[:,:3]


class Tstep(object):
    """

    """
    def __init__(self):
        self.kernel = ext.p2p_tstep_kernel
        self.kernel.local_size = 384
        self.kernel.set_arg('LMEM', 6, 8)

    def set_arg(self, mode, i, arg):
        self.kernel.set_arg(mode, i, arg)

    @property
    def global_size(self, value):
        self.kernel.global_size = value

    def run(self, ni):
        self.kernel.global_size = ni
        self.kernel.run()
        return self

    def get_result(self):
        result = self.kernel.get_result()[0]
        return result




@decallmethods(timings)
class Gravity(object):
    """
    A base class for gravitational interaction between particles.
    """
    def __init__(self):
        self.phi_kernel = ext.p2p_phi_kernel
        self.phi_kernel.local_size = 384
        self.phi_kernel.set_arg('LMEM', 5, 8)

        self.acc_kernel = ext.p2p_acc_kernel
        self.acc_kernel.local_size = 384
        self.acc_kernel.set_arg('LMEM', 5, 8)

        self.acc_jerk_kernel = ext.p2p_acc_jerk_kernel
        self.acc_jerk_kernel.local_size = 384
        self.acc_jerk_kernel.set_arg('LMEM', 5, 8)

        self.tstep_kernel = ext.p2p_tstep_kernel
        self.tstep_kernel.local_size = 384
        self.tstep_kernel.set_arg('LMEM', 6, 8)

        self.pnacc_kernel = ext.p2p_pnacc_kernel
        self.pnacc_kernel.local_size = 384
        self.pnacc_kernel.set_arg('LMEM', 13, 8)


    ### phi methods

    def setup_phi_data(self, ni, idata, nj, jdata):
        self.phi_kernel.set_arg('IN', 0, ni)
        self.phi_kernel.set_arg('IN', 1, idata)
        self.phi_kernel.set_arg('IN', 2, nj)
        self.phi_kernel.set_arg('IN', 3, jdata)
        self.phi_kernel.set_arg('OUT', 4, (ni,))

    def get_phi(self, ni, idata, nj, jdata):
        """
        Get obj-obj newtonian phi.
        """
        self.setup_phi_data(ni, idata, nj, jdata)

        phi_kernel = self.phi_kernel
        phi_kernel.global_size = ni
        phi_kernel.run()
        result = phi_kernel.get_result()[0]
        return result


    ### acc methods

    def setup_acc_data(self, ni, idata, nj, jdata):
        self.acc_kernel.set_arg('IN', 0, ni)
        self.acc_kernel.set_arg('IN', 1, idata)
        self.acc_kernel.set_arg('IN', 2, nj)
        self.acc_kernel.set_arg('IN', 3, jdata)
        self.acc_kernel.set_arg('OUT', 4, (ni, 4))          # XXX: forcing shape = (ni, 4) due to
                                                            #      a bug using __global REAL3 in
                                                            #      AMD's OpenCL implementation.

    def get_acc(self, ni, idata, nj, jdata):
        """
        Get obj-obj newtonian acc.
        """
        self.setup_acc_data(ni, idata, nj, jdata)

        acc_kernel = self.acc_kernel
        acc_kernel.global_size = ni
        acc_kernel.run()
        result = acc_kernel.get_result()[0]
        return result[:,:3]                                 # XXX: forcing return shape = (ni, 3).
                                                            #      see comment about a bug using
                                                            #      __global REAL3 in OpenCL.


    ### acc-jerk methods

    def setup_acc_jerk_data(self, ni, idata, nj, jdata):
        self.acc_jerk_kernel.set_arg('IN', 0, ni)
        self.acc_jerk_kernel.set_arg('IN', 1, idata)
        self.acc_jerk_kernel.set_arg('IN', 2, nj)
        self.acc_jerk_kernel.set_arg('IN', 3, jdata)
        self.acc_jerk_kernel.set_arg('OUT', 4, (ni, 8))
#        self.acc_jerk_kernel.set_arg('OUT', 5, (ni, 4))     # XXX: forcing shape = (ni, 4) due to
                                                            #      a bug using __global REAL3 in
                                                            #      AMD's OpenCL implementation.

    def get_acc_jerk(self, ni, idata, nj, jdata):
        """
        Get obj-obj newtonian acc.
        """
        self.setup_acc_jerk_data(ni, idata, nj, jdata)

        acc_jerk_kernel = self.acc_jerk_kernel
        acc_jerk_kernel.global_size = ni
        acc_jerk_kernel.run()
        result = acc_jerk_kernel.get_result()[0]
        return (result[:,:3], result[:,4:7])
#        return (result[0][:,:3], result[1][:,:3])           # XXX: forcing return shape = (ni, 3).
                                                            #      see comment about a bug using
                                                            #      __global REAL3 in OpenCL.


    ### tstep methods

    def setup_tstep_data(self, ni, idata, nj, jdata, eta):
        self.tstep_kernel.set_arg('IN', 0, ni)
        self.tstep_kernel.set_arg('IN', 1, idata)
        self.tstep_kernel.set_arg('IN', 2, nj)
        self.tstep_kernel.set_arg('IN', 3, jdata)
        self.tstep_kernel.set_arg('IN', 4, eta)
        self.tstep_kernel.set_arg('OUT', 5, (ni,))

    def get_tstep(self, ni, idata, nj, jdata, eta):
        """
        Get timestep.
        """
        self.setup_tstep_data(ni, idata, nj, jdata, eta)

        tstep_kernel = self.tstep_kernel
        tstep_kernel.global_size = ni
        tstep_kernel.run()
        result = tstep_kernel.get_result()[0]
        return result


    ### pnacc methods

    def setup_pnacc_data(self, ni, idata, nj, jdata, pn_order, clight):
        clight = Clight(pn_order, clight)
        self.pnacc_kernel.set_arg('IN', 0, ni)
        self.pnacc_kernel.set_arg('IN', 1, idata)
        self.pnacc_kernel.set_arg('IN', 2, nj)
        self.pnacc_kernel.set_arg('IN', 3, jdata)
        self.pnacc_kernel.set_arg('IN', 4, clight.pn_order)
        self.pnacc_kernel.set_arg('IN', 5, clight.inv1)
        self.pnacc_kernel.set_arg('IN', 6, clight.inv2)
        self.pnacc_kernel.set_arg('IN', 7, clight.inv3)
        self.pnacc_kernel.set_arg('IN', 8, clight.inv4)
        self.pnacc_kernel.set_arg('IN', 9, clight.inv5)
        self.pnacc_kernel.set_arg('IN', 10, clight.inv6)
        self.pnacc_kernel.set_arg('IN', 11, clight.inv7)
        self.pnacc_kernel.set_arg('OUT', 12, (ni, 4))       # XXX: forcing shape = (ni, 4) due to
                                                            #      a bug using __global REAL3 in
                                                            #      AMD's OpenCL implementation.

    def get_pnacc(self, ni, idata, nj, jdata, pn_order, clight):
        """
        Get blackhole-blackhole post-newtonian acc.
        """
        self.setup_pnacc_data(ni, idata, nj, jdata, pn_order, clight)

        pnacc_kernel = self.pnacc_kernel
        pnacc_kernel.global_size = ni
        pnacc_kernel.run()
        result = pnacc_kernel.get_result()[0]
        return result[:,:3]                                 # XXX: forcing return shape = (ni, 3).
                                                            #      see comment about a bug using
                                                            #      __global REAL3 in OpenCL.



gravitation = Gravity()

#phi = Phi()
#acc = Acc()
#tstep = Tstep()

phi = ext.p2p_phi_kernel
phi.local_size = 384
phi.set_arg('LMEM', 5, 8)

acc = ext.p2p_acc_kernel
acc.local_size = 384
acc.set_arg('LMEM', 5, 8)

tstep = ext.p2p_tstep_kernel
tstep.local_size = 384
tstep.set_arg('LMEM', 6, 8)

pnacc = ext.p2p_pnacc_kernel
pnacc.local_size = 384
pnacc.set_arg('LMEM', 13, 8)

acc_jerk = ext.p2p_acc_jerk_kernel
acc_jerk.local_size = 384
acc_jerk.set_arg('LMEM', 5, 8)


########## end of file ##########
