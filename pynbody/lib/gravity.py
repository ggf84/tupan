#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements a minimal class for gravitational interactions
between particles in Newtonian and post-Newtonian approach.
"""


import sys
import numpy as np
from .extensions import kernel_library
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
        pass


    ### phi methods

    def setup_phi_data(self, ni, ipos, imass, ieps2,
                             nj, jpos, jmass, jeps2):

        iposmass = np.concatenate((ipos, imass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jpos, jmass[..., np.newaxis]), axis=1)

        data = (ni, iposmass, ieps2,
                nj, jposmass, jeps2)
        return data

    def set_phi(self, ni, ipos, imass, ieps2,
                      nj, jpos, jmass, jeps2):
        """
        Set obj-obj newtonian phi.
        """
        data = self.setup_phi_data(ni, ipos, imass, ieps2,
                                   nj, jpos, jmass, jeps2)

        result_shape = (ni,)
        local_memory_shape = (4, 1)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        phi_kernel = kernel_library.get_kernel("p2p_phi_kernel")

        phi_kernel.set_kernel_args(*data, global_size=global_size,
                                          local_size=local_size,
                                          result_shape=result_shape,
                                          local_memory_shape=local_memory_shape)
        phi_kernel.run()
        ret = phi_kernel.get_result()
        return ret


    ### acc methods

    def setup_acc_data(self, ni, ipos, imass, ieps2,
                             nj, jpos, jmass, jeps2):

        iposmass = np.concatenate((ipos, imass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jpos, jmass[..., np.newaxis]), axis=1)

        data = (ni, iposmass, ieps2,
                nj, jposmass, jeps2)
        return data

    def set_acc(self, ni, ipos, imass, ieps2,
                      nj, jpos, jmass, jeps2):
        """
        Set obj-obj newtonian acc.
        """
        data = self.setup_acc_data(ni, ipos, imass, ieps2,
                                   nj, jpos, jmass, jeps2)

        result_shape = (ni, 4)          # XXX: forcing shape = (ni, 4) due to
                                        #      a bug using __global REAL3 in
                                        #      AMD's OpenCL implementation.
        local_memory_shape = (4, 1)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        acc_kernel = kernel_library.get_kernel("p2p_acc_kernel")

        acc_kernel.set_kernel_args(*data, global_size=global_size,
                                          local_size=local_size,
                                          result_shape=result_shape,
                                          local_memory_shape=local_memory_shape)
        acc_kernel.run()
        ret = acc_kernel.get_result()
        return ret[:,:3]                # XXX: forcing return shape = (ni, 3).
                                        #      see comment about a bug using
                                        #      __global REAL3 in OpenCL.


    ### tstep methods

    def setup_tstep_data(self, ni, ipos, imass, ivel, ieps2,
                               nj, jpos, jmass, jvel, jeps2, eta):

        iposmass = np.concatenate((ipos, imass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jpos, jmass[..., np.newaxis]), axis=1)
        iveleps2 = np.concatenate((ivel, ieps2[..., np.newaxis]), axis=1)
        jveleps2 = np.concatenate((jvel, jeps2[..., np.newaxis]), axis=1)

        data = (ni, iposmass, iveleps2,
                nj, jposmass, jveleps2,
                eta)
        return data

    def set_tstep(self, ni, ipos, imass, ivel, ieps2,
                        nj, jpos, jmass, jvel, jeps2, eta):
        """
        Set timestep.
        """
        data = self.setup_tstep_data(ni, ipos, imass, ivel, ieps2,
                                     nj, jpos, jmass, jvel, jeps2, eta)

        result_shape = (ni,)
        local_memory_shape = (4, 4)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        tstep_kernel = kernel_library.get_kernel("p2p_tstep_kernel")

        tstep_kernel.set_kernel_args(*data, global_size=global_size,
                                            local_size=local_size,
                                            result_shape=result_shape,
                                            local_memory_shape=local_memory_shape)
        tstep_kernel.run()
        ret = tstep_kernel.get_result()
        return ret


    ### pnacc methods

    def setup_pnacc_data(self, ni, ipos, imass, ivel,
                               nj, jpos, jmass, jvel,
                               pn_order, clight):

        iposmass = np.concatenate((ipos, imass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jpos, jmass[..., np.newaxis]), axis=1)
        iveliv2 = np.concatenate((ivel, (ivel**2).sum(1)[..., np.newaxis]), axis=1)
        jveljv2 = np.concatenate((jvel, (jvel**2).sum(1)[..., np.newaxis]), axis=1)
        clight = Clight(pn_order, clight)

        data = (ni, iposmass, iveliv2,
                nj, jposmass, jveljv2,
                clight.pn_order, clight.inv1,
                clight.inv2, clight.inv3,
                clight.inv4, clight.inv5,
                clight.inv6, clight.inv7)
        return data

    def set_pnacc(self, ni, ipos, imass, ivel,
                        nj, jpos, jmass, jvel,
                        pn_order, clight):
        """
        Set blackhole-blackhole post-newtonian acc.
        """
        data = self.setup_pnacc_data(ni, ipos, imass, ivel,
                                     nj, jpos, jmass, jvel,
                                     pn_order, clight)

        result_shape = (ni,4)           # XXX: forcing shape = (ni, 4) due to
                                        #      a bug using __global REAL3 in
                                        #      AMD's OpenCL implementation.
        local_memory_shape = (4, 4)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        pnacc_kernel = kernel_library.get_kernel("p2p_pnacc_kernel")

        pnacc_kernel.set_kernel_args(*data, global_size=global_size,
                                            local_size=local_size,
                                            result_shape=result_shape,
                                            local_memory_shape=local_memory_shape)
        pnacc_kernel.run()
        ret = pnacc_kernel.get_result()
        return ret[:,:3]                # XXX: forcing return shape = (ni, 3).
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
