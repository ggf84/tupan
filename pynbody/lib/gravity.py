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


@decallmethods(timings)
class Gravity(object):
    """
    A base class for gravitational interaction between particles.
    """
    def __init__(self):
        self.phi_kernel = ext.p2p_phi_kernel
        self.phi_kernel.local_size = 384
        self.phi_kernel.set_arg('LMEM', 7, 4)
        self.phi_kernel.set_arg('LMEM', 8, 1)

        self.acc_kernel = ext.p2p_acc_kernel
        self.acc_kernel.local_size = 384
        self.acc_kernel.set_arg('LMEM', 7, 4)
        self.acc_kernel.set_arg('LMEM', 8, 1)

        self.tstep_kernel = ext.p2p_tstep_kernel
        self.tstep_kernel.local_size = 384
        self.tstep_kernel.set_arg('LMEM', 6, 8)

        self.pnacc_kernel = ext.p2p_pnacc_kernel
        self.pnacc_kernel.local_size = 384
        self.pnacc_kernel.set_arg('LMEM', 15, 4)
        self.pnacc_kernel.set_arg('LMEM', 16, 4)


    ### phi methods

    def setup_phi_data(self, ni, iposmass, ieps2,
                             nj, jposmass, jeps2):
        self.phi_kernel.set_arg('IN', 0, ni)
        self.phi_kernel.set_arg('IN', 1, iposmass)
        self.phi_kernel.set_arg('IN', 2, ieps2)
        self.phi_kernel.set_arg('IN', 3, nj)
        self.phi_kernel.set_arg('IN', 4, jposmass)
        self.phi_kernel.set_arg('IN', 5, jeps2)
        self.phi_kernel.set_arg('OUT', 6, (ni,))

    def set_phi(self, ni, iposmass, ieps2,
                      nj, jposmass, jeps2):
        """
        Set obj-obj newtonian phi.
        """
        self.setup_phi_data(ni, iposmass, ieps2,
                            nj, jposmass, jeps2)

        phi_kernel = self.phi_kernel
        phi_kernel.global_size = ni
        phi_kernel.run()
        result = phi_kernel.get_result()[0]
        return result


    ### acc methods

    def setup_acc_data(self, ni, iposmass, ieps2,
                             nj, jposmass, jeps2):
        self.acc_kernel.set_arg('IN', 0, ni)
        self.acc_kernel.set_arg('IN', 1, iposmass)
        self.acc_kernel.set_arg('IN', 2, ieps2)
        self.acc_kernel.set_arg('IN', 3, nj)
        self.acc_kernel.set_arg('IN', 4, jposmass)
        self.acc_kernel.set_arg('IN', 5, jeps2)
        self.acc_kernel.set_arg('OUT', 6, (ni, 4))      # XXX: forcing shape = (ni, 4) due to
                                                        #      a bug using __global REAL3 in
                                                        #      AMD's OpenCL implementation.

    def set_acc(self, ni, iposmass, ieps2,
                      nj, jposmass, jeps2):
        """
        Set obj-obj newtonian acc.
        """
        self.setup_acc_data(ni, iposmass, ieps2,
                            nj, jposmass, jeps2)

        acc_kernel = self.acc_kernel
        acc_kernel.global_size = ni
        acc_kernel.run()
        result = acc_kernel.get_result()[0]
        return result[:,:3]             # XXX: forcing return shape = (ni, 3).
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

    def set_tstep(self, ni, idata, nj, jdata, eta):
        """
        Set timestep.
        """
        self.setup_tstep_data(ni, idata, nj, jdata, eta)

        tstep_kernel = self.tstep_kernel
        tstep_kernel.global_size = ni
        tstep_kernel.run()
        result = tstep_kernel.get_result()[0]
        return result


    ### pnacc methods

    def setup_pnacc_data(self, ni, ipos, imass, ivel,
                               nj, jpos, jmass, jvel,
                               pn_order, clight):

        iposmass = np.concatenate((ipos, imass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jpos, jmass[..., np.newaxis]), axis=1)
        iveliv2 = np.concatenate((ivel, (ivel**2).sum(1)[..., np.newaxis]), axis=1)
        jveljv2 = np.concatenate((jvel, (jvel**2).sum(1)[..., np.newaxis]), axis=1)
        clight = Clight(pn_order, clight)

        self.pnacc_kernel.set_arg('IN', 0, ni)
        self.pnacc_kernel.set_arg('IN', 1, iposmass)
        self.pnacc_kernel.set_arg('IN', 2, iveliv2)
        self.pnacc_kernel.set_arg('IN', 3, nj)
        self.pnacc_kernel.set_arg('IN', 4, jposmass)
        self.pnacc_kernel.set_arg('IN', 5, jveljv2)
        self.pnacc_kernel.set_arg('IN', 6, clight.pn_order)
        self.pnacc_kernel.set_arg('IN', 7, clight.inv1)
        self.pnacc_kernel.set_arg('IN', 8, clight.inv2)
        self.pnacc_kernel.set_arg('IN', 9, clight.inv3)
        self.pnacc_kernel.set_arg('IN', 10, clight.inv4)
        self.pnacc_kernel.set_arg('IN', 11, clight.inv5)
        self.pnacc_kernel.set_arg('IN', 12, clight.inv6)
        self.pnacc_kernel.set_arg('IN', 13, clight.inv7)
        self.pnacc_kernel.set_arg('OUT', 14, (ni, 4))       # XXX: forcing shape = (ni, 4) due to
                                                            #      a bug using __global REAL3 in
                                                            #      AMD's OpenCL implementation.

    def set_pnacc(self, ni, ipos, imass, ivel,
                        nj, jpos, jmass, jvel,
                        pn_order, clight):
        """
        Set blackhole-blackhole post-newtonian acc.
        """
        self.setup_pnacc_data(ni, ipos, imass, ivel,
                              nj, jpos, jmass, jvel,
                              pn_order, clight)

        pnacc_kernel = self.pnacc_kernel
        pnacc_kernel.global_size = ni
        pnacc_kernel.run()
        result = pnacc_kernel.get_result()[0]
        return result[:,:3]             # XXX: forcing return shape = (ni, 3).
                                        #      see comment about a bug using
                                        #      __global REAL3 in OpenCL.





    ### acctstep methods

    def setup_acctstep_data(self, iobj, jobj, eta):   # XXX: deprecated!
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
        iveleps2 = np.concatenate((iobj.vel, iobj.eps2[..., np.newaxis]), axis=1)
        jveleps2 = np.concatenate((jobj.vel, jobj.eps2[..., np.newaxis]), axis=1)
        data = (iposmass, iveleps2,
                jposmass, jveleps2,
                np.uint32(ni),
                np.uint32(nj),
                np.float64(eta))
        return data

    def set_acctstep(self, iobj, jobj, eta):   # XXX: deprecated!
        """
        Set obj-obj newtonian acc and timestep.
        """
        ni = len(iobj)
        data = self.setup_acctstep_data(iobj, jobj, eta)

        result_shape = (ni, 4)
        local_memory_shape = (4, 4)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        acc_kernel = kernel_library.get_kernel("p2p_acctstep_kernel")

        acc_kernel.set_kernel_args(*data, global_size=global_size,
                                          local_size=local_size,
                                          result_shape=result_shape,
                                          local_memory_shape=local_memory_shape)
        acc_kernel.run()
        ret = acc_kernel.get_result()
        return (ret[:,:3], ret[:,3])



gravitation = Gravity()


########## end of file ##########
