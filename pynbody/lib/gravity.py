#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements a minimal class for gravitational interactions
between particles in Newtonian and post-Newtonian approach.
"""


import sys
import numpy as np
from collections import namedtuple
from .extensions import kernel_library
from .utils.timing import timings


__all__ = ["Gravity", "gravitation"]


class Newtonian(object):
    """
    This class holds base methods for newtonian gravity.
    """
    def __init__(self):
        pass


    def set_phi(self, iobj, jobj):
        """
        Set obj-obj newtonian phi.
        """
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        data = (iposmass, iobj.eps2,
                jposmass, jobj.eps2,
                np.uint32(ni),
                np.uint32(nj))

        output_buf = np.empty(ni)
        lmem_layout = (4, 1)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        phi_kernel = kernel_library.get_kernel("p2p_phi_kernel")

        phi_kernel.set_kernel_args(*data, global_size=global_size,
                                          local_size=local_size,
                                          output_buf=output_buf,
                                          lmem_layout=lmem_layout)
        phi_kernel.run()
        ret = phi_kernel.get_result()
        return ret


    def set_acc(self, iobj, jobj):
        """
        Set obj-obj newtonian acc.
        """
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        data = (iposmass, iobj.eps2,
                jposmass, jobj.eps2,
                np.uint32(ni),
                np.uint32(nj))

        output_buf = np.empty((ni,4))   # XXX: forcing shape = (ni, 4) due to
                                        #      a bug using __global REAL3 in
                                        #      AMD's OpenCL implementation.
        lmem_layout = (4, 1)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        acc_kernel = kernel_library.get_kernel("p2p_acc_kernel")

        acc_kernel.set_kernel_args(*data, global_size=global_size,
                                          local_size=local_size,
                                          output_buf=output_buf,
                                          lmem_layout=lmem_layout)
        acc_kernel.run()
        ret = acc_kernel.get_result()
        return ret[:,:3]                # XXX: forcing return shape = (ni, 3).
                                        #      see comment about a bug using
                                        #      __global REAL3 in OpenCL.


    def set_acctstep(self, iobj, jobj, tau):
        """
        Set obj-obj newtonian acc and timestep.
        """
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        iveleps2 = np.vstack((iobj.vel.T, iobj.eps2)).T
        jveleps2 = np.vstack((jobj.vel.T, jobj.eps2)).T
        data = (iposmass, iveleps2,
                jposmass, jveleps2,
                np.uint32(ni),
                np.uint32(nj),
                np.float64(tau))

        output_buf = np.empty((ni,4))
        lmem_layout = (4, 4)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        acc_kernel = kernel_library.get_kernel("p2p_acctstep_kernel")

        acc_kernel.set_kernel_args(*data, global_size=global_size,
                                          local_size=local_size,
                                          output_buf=output_buf,
                                          lmem_layout=lmem_layout)
        acc_kernel.run()
        ret = acc_kernel.get_result()
        return (ret[:,:3], ret[:,3])


    def set_tstep(self, iobj, jobj, tau):
        """
        Set timestep.
        """
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        iveleps2 = np.vstack((iobj.vel.T, iobj.eps2)).T
        jveleps2 = np.vstack((jobj.vel.T, jobj.eps2)).T
        data = (iposmass, iveleps2,
                jposmass, jveleps2,
                np.uint32(ni),
                np.uint32(nj),
                np.float64(tau))

        output_buf = np.empty(ni)
        lmem_layout = (4, 4)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        tstep_kernel = kernel_library.get_kernel("p2p_tstep_kernel")

        tstep_kernel.set_kernel_args(*data, global_size=global_size,
                                            local_size=local_size,
                                            output_buf=output_buf,
                                            lmem_layout=lmem_layout)
        tstep_kernel.run()
        ret = tstep_kernel.get_result()
        return ret


class PostNewtonian(object):
    """
    This class holds base methods for post-newtonian gravity.
    """
    def __init__(self, pn_order, clight):
        self.clight = Clight(pn_order, clight)


    def set_acc(self, iobj, jobj):
        """
        Set blackhole-blackhole post-newtonian acc.
        """
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        iveliv2 = np.vstack((iobj.vel.T, (iobj.vel**2).sum(1))).T
        jveljv2 = np.vstack((jobj.vel.T, (jobj.vel**2).sum(1))).T
        clight = self.clight
        data = (iposmass, iveliv2,
                jposmass, jveljv2,
                np.uint32(ni),
                np.uint32(nj),
                np.uint32(clight.pn_order), np.float64(clight.inv1),
                np.float64(clight.inv2), np.float64(clight.inv3),
                np.float64(clight.inv4), np.float64(clight.inv5),
                np.float64(clight.inv6), np.float64(clight.inv7),
               )

        output_buf = np.empty((ni,4))   # XXX: forcing shape = (ni, 4) due to
                                        #      a bug using __global REAL3 in
                                        #      AMD's OpenCL implementation.
        lmem_layout = (4, 4)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        pnacc_kernel = kernel_library.get_kernel("p2p_pnacc_kernel")

        pnacc_kernel.set_kernel_args(*data, global_size=global_size,
                                            local_size=local_size,
                                            output_buf=output_buf,
                                            lmem_layout=lmem_layout)
        pnacc_kernel.run()
        ret = pnacc_kernel.get_result()
        return ret[:,:3]                # XXX: forcing return shape = (ni, 3).
                                        #      see comment about a bug using
                                        #      __global REAL3 in OpenCL.


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


class Gravity(object):
    """
    A base class for gravitational interaction between particles.
    """
    def __init__(self, pn_order=4, clight=25.0):
        self.newtonian = Newtonian()
        self.post_newtonian = PostNewtonian(pn_order, clight)


gravitation = Gravity()


########## end of file ##########
