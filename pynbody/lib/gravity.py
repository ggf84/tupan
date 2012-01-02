#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements a minimal class for gravitational interactions
between particles in Newtonian and post-Newtonian approach.
"""


import sys
import numpy as np
from collections import namedtuple
from . import extensions
from .utils.timing import timings


__all__ = ["Gravity", "gravity_methods"]


class Newtonian(object):
    """
    This class holds base methods for newtonian gravity.
    """
    def __init__(self, set_phi, set_acc):
        self.set_phi = set_phi
        self.set_acc = set_acc


class PostNewtonian(object):
    """
    This class holds base methods for post-newtonian gravity.
    """
    def __init__(self, set_acc):
        self.set_acc = set_acc


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
        self._clight = Clight(pn_order, clight)
        self.newtonian = None
        self.post_newtonian = None
        self._has_built = False


    def _setup(self):
        kernels = extensions.build_kernels()
        self._phi_kernel = kernels["cl_lib64_p2p_phi_kernel"]
        self._acc_kernel = kernels["cl_lib64_p2p_acc_kernel"]
        self._pnacc_kernel = kernels["c_lib64_p2p_pnacc_kernel"]


    def build(self):
        if not self._has_built:
            self._setup()
            self.newtonian = Newtonian(self._set_phi, self._set_acc)
            self.post_newtonian = PostNewtonian(self._set_pnacc)
            self._has_built = True


    def _set_phi(self, iobj, jobj):
        """
        Set obj-obj phi.
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

        self._phi_kernel.load_data(*data, global_size=global_size,
                                          local_size=local_size,
                                          output_buf=output_buf,
                                          lmem_layout=lmem_layout)
        self._phi_kernel.execute()
        ret = self._phi_kernel.get_result()
        return ret


    def _set_acc(self, iobj, jobj, eta):
        """
        Set obj-obj acc.
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
                np.float64(eta))

        output_buf = np.empty((ni,4))
        lmem_layout = (4, 4)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        self._acc_kernel.load_data(*data, global_size=global_size,
                                          local_size=local_size,
                                          output_buf=output_buf,
                                          lmem_layout=lmem_layout)
        self._acc_kernel.execute()
        ret = self._acc_kernel.get_result()
        return (ret[:,:3], ret[:,3])


    def _set_pnacc(self, iobj, jobj):
        """
        Set blackhole-blackhole pn-acc.
        """
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        clight = self._clight
        data = (iposmass, iobj.vel,
                jposmass, jobj.vel,
                np.uint32(ni),
                np.uint32(nj),
                np.uint32(clight.pn_order), np.float64(clight.inv1),
                np.float64(clight.inv2), np.float64(clight.inv3),
                np.float64(clight.inv4), np.float64(clight.inv5),
                np.float64(clight.inv6), np.float64(clight.inv7)
               )
        self._pnacc_kernel.load_data(*data)
        self._pnacc_kernel.execute()
        ret = self._pnacc_kernel.get_result()
        return ret


gravity_methods = Gravity()


########## end of file ##########
