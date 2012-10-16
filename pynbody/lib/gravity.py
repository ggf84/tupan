#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements a minimal class for gravitational interactions
between particles in Newtonian and post-Newtonian approach.
"""


import numpy as np
from .extensions import kernels
from .utils.timing import decallmethods, timings


__all__ = ["Phi", "phi",
           "Acc", "acc",
           "Tstep", "tstep",
           "PNAcc", "pnacc",
           "AccJerk", "acc_jerk"]


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



@decallmethods(timings)
class Phi(object):
    """

    """
    def __init__(self, libgrav):
        self.kernel = libgrav.p2p_phi_kernel
        self.kernel.local_size = 384
        self.output = np.zeros(0, dtype=self.kernel.env.dtype)


    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n
        idata = np.concatenate((iobj.pos.T, iobj.mass.reshape(1,-1),
                                iobj.vel.T, iobj.eps2.reshape(1,-1))).T
        jdata = np.concatenate((jobj.pos.T, jobj.mass.reshape(1,-1),
                                jobj.vel.T, jobj.eps2.reshape(1,-1))).T

        if ni > len(self.output):
            self.output = np.zeros(ni, dtype=self.kernel.env.dtype)

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_input_buffer(1, idata)
        self.kernel.set_int(2, nj)
        self.kernel.set_input_buffer(3, jdata)
        self.kernel.set_output_buffer(4, self.output[:ni])
        self.kernel.set_local_memory(5, 8)


    def run(self):
        self.kernel.run()


    def get_result(self):
        return self.kernel.get_result()[0]



@decallmethods(timings)
class Acc(object):
    """

    """
    def __init__(self, libgrav):
        self.kernel = libgrav.p2p_acc_kernel
        self.kernel.local_size = 384
        self.output = np.zeros((0, 4), dtype=self.kernel.env.dtype)


    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n
        idata = np.concatenate((iobj.pos.T, iobj.mass.reshape(1,-1),
                                iobj.vel.T, iobj.eps2.reshape(1,-1))).T
        jdata = np.concatenate((jobj.pos.T, jobj.mass.reshape(1,-1),
                                jobj.vel.T, jobj.eps2.reshape(1,-1))).T

        if ni > len(self.output):
            self.output = np.zeros((ni, 4), dtype=self.kernel.env.dtype)

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_input_buffer(1, idata)
        self.kernel.set_int(2, nj)
        self.kernel.set_input_buffer(3, jdata)
        self.kernel.set_output_buffer(4, self.output[:ni])
        self.kernel.set_local_memory(5, 8)


    def run(self):
        self.kernel.run()


    def get_result(self):
        return self.kernel.get_result()[0][:,:3]



@decallmethods(timings)
class Tstep(object):
    """

    """
    def __init__(self, libgrav):
        self.kernel = libgrav.p2p_tstep_kernel
        self.kernel.local_size = 384
        self.output = np.zeros(0, dtype=self.kernel.env.dtype)


    def set_args(self, iobj, jobj, eta):
        ni = iobj.n
        nj = jobj.n
        idata = np.concatenate((iobj.pos.T, iobj.mass.reshape(1,-1),
                                iobj.vel.T, iobj.eps2.reshape(1,-1))).T
        jdata = np.concatenate((jobj.pos.T, jobj.mass.reshape(1,-1),
                                jobj.vel.T, jobj.eps2.reshape(1,-1))).T

        if ni > len(self.output):
            self.output = np.zeros(ni, dtype=self.kernel.env.dtype)

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_input_buffer(1, idata)
        self.kernel.set_int(2, nj)
        self.kernel.set_input_buffer(3, jdata)
        self.kernel.set_float(4, eta)
        self.kernel.set_output_buffer(5, self.output[:ni])
        self.kernel.set_local_memory(6, 8)


    def run(self):
        self.kernel.run()


    def get_result(self):
        return self.kernel.get_result()[0]



@decallmethods(timings)
class PNAcc(object):
    """

    """
    def __init__(self, libgrav):
        self.kernel = libgrav.p2p_pnacc_kernel
        self.kernel.local_size = 384
        self.output = np.zeros((0, 4), dtype=self.kernel.env.dtype)


    def set_args(self, iobj, jobj, pn_order, clight):
        ni = iobj.n
        nj = jobj.n
        idata = np.concatenate((iobj.pos.T, iobj.mass.reshape(1,-1),
                                iobj.vel.T, iobj.eps2.reshape(1,-1))).T
        jdata = np.concatenate((jobj.pos.T, jobj.mass.reshape(1,-1),
                                jobj.vel.T, jobj.eps2.reshape(1,-1))).T

        clight = Clight(pn_order, clight)

        if ni > len(self.output):
            self.output = np.zeros((ni, 4), dtype=self.kernel.env.dtype)

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_input_buffer(1, idata)
        self.kernel.set_int(2, nj)
        self.kernel.set_input_buffer(3, jdata)
        self.kernel.set_int(4, clight.pn_order)
        self.kernel.set_float(5, clight.inv1)
        self.kernel.set_float(6, clight.inv2)
        self.kernel.set_float(7, clight.inv3)
        self.kernel.set_float(8, clight.inv4)
        self.kernel.set_float(9, clight.inv5)
        self.kernel.set_float(10, clight.inv6)
        self.kernel.set_float(11, clight.inv7)
        self.kernel.set_output_buffer(12, self.output[:ni])
        self.kernel.set_local_memory(13, 8)


    def run(self):
        self.kernel.run()


    def get_result(self):
        return self.kernel.get_result()[0][:,:3]



@decallmethods(timings)
class AccJerk(object):
    """

    """
    def __init__(self, libgrav):
        self.kernel = libgrav.p2p_acc_jerk_kernel
        self.kernel.local_size = 384
        self.output = np.zeros((0, 8), dtype=self.kernel.env.dtype)


    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n
        idata = np.concatenate((iobj.pos.T, iobj.mass.reshape(1,-1),
                                iobj.vel.T, iobj.eps2.reshape(1,-1))).T
        jdata = np.concatenate((jobj.pos.T, jobj.mass.reshape(1,-1),
                                jobj.vel.T, jobj.eps2.reshape(1,-1))).T

        if ni > len(self.output):
            self.output = np.zeros((ni, 8), dtype=self.kernel.env.dtype)

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_input_buffer(1, idata)
        self.kernel.set_int(2, nj)
        self.kernel.set_input_buffer(3, jdata)
        self.kernel.set_output_buffer(4, self.output[:ni])
        self.kernel.set_local_memory(5, 8)


    def run(self):
        self.kernel.run()


    def get_result(self):
        result = self.kernel.get_result()[0]
        return (result[:,:3], result[:,4:7])



phi = Phi(kernels)
acc = Acc(kernels)
tstep = Tstep(kernels)
pnacc = PNAcc(kernels)
acc_jerk = AccJerk(kernels)


########## end of file ##########
