#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements a minimal class for gravitational interactions between
different particle types at Newtonian and post-Newtonian approach.
"""


import sys
import numpy as np
from collections import namedtuple
from .utils.timing import timings
from . import extensions


__all__ = ["Gravity", "gravity_kernels"]


class Clight(object):
    """

    """
    def __init__(self, pn_order, clight):
        self.pn_order = int(pn_order)
        self._clight = float(clight)
        self.inv1 = 1.0/self._clight
        self.inv2 = self.inv1**2
        self.inv3 = self.inv1**3
        self.inv4 = self.inv1**4
        self.inv5 = self.inv1**5
        self.inv6 = self.inv1**6
        self.inv7 = self.inv1**7


class Newtonian(object):
    """
    A base class for newtonian gravity.
    """
    def __init__(self, acc_kernel, phi_kernel):
        self._acc_kernel = acc_kernel
        self._phi_kernel = phi_kernel


    def _set_acc(self, iobj, jobj):
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        data = (np.uint32(ni),
                np.uint32(nj),
                iposmass, iobj.eps2,
                jposmass, jobj.eps2)

        output_buf = np.empty((ni,4))

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        self._acc_kernel.load_data(*data, global_size=global_size,
                                     local_size=local_size,
                                     output_buf=output_buf)
        self._acc_kernel.execute()
        ret = self._acc_kernel.get_result()
        return (ret[:,:3], ret[:,3])


    def _set_phi(self, iobj, jobj):
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        data = (np.uint32(ni),
                np.uint32(nj),
                iposmass, iobj.eps2,
                jposmass, jobj.eps2)

        output_buf = np.empty(ni)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        self._phi_kernel.load_data(*data, global_size=global_size,
                                     local_size=local_size,
                                     output_buf=output_buf)
        self._phi_kernel.execute()
        ret = self._phi_kernel.get_result()
        return ret


    # body-body
    def set_acc_b2b(self, iobj, jobj):
        """
        Set body-body acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_b2b(self, iobj, jobj):
        """
        Set body-body phi.
        """
        return self._set_phi(iobj, jobj)

    # body-blackhole
    def set_acc_b2bh(self, iobj, jobj):
        """
        Set body-blackhole acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_b2bh(self, iobj, jobj):
        """
        Set body-blackhole phi.
        """
        return self._set_phi(iobj, jobj)

    # body-sph
    def set_acc_b2sph(self, iobj, jobj):
        """
        Set body-sph acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_b2sph(self, iobj, jobj):
        """
        Set body-sph phi.
        """
        return self._set_phi(iobj, jobj)


    # blackhole-blackhole
    def set_acc_bh2bh(self, iobj, jobj):
        """
        Set blackhole-blackhole acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_bh2bh(self, iobj, jobj):
        """
        Set blackhole-blackhole phi.
        """
        return self._set_phi(iobj, jobj)

    # blackhole-body
    def set_acc_bh2b(self, iobj, jobj):
        """
        Set blackhole-body acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_bh2b(self, iobj, jobj):
        """
        Set blackhole-body phi.
        """
        return self._set_phi(iobj, jobj)

    # blackhole-sph
    def set_acc_bh2sph(self, iobj, jobj):
        """
        Set blackhole-sph acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_bh2sph(self, iobj, jobj):
        """
        Set blackhole-sph phi.
        """
        return self._set_phi(iobj, jobj)


    # sph-sph
    def set_acc_sph2sph(self, iobj, jobj):
        """
        Set sph-sph acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_sph2sph(self, iobj, jobj):
        """
        Set sph-sph phi.
        """
        return self._set_phi(iobj, jobj)

    # sph-body
    def set_acc_sph2b(self, iobj, jobj):
        """
        Set sph-body acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_sph2b(self, iobj, jobj):
        """
        Set sph-body phi.
        """
        return self._set_phi(iobj, jobj)

    # sph-blackhole
    def set_acc_sph2bh(self, iobj, jobj):
        """
        Set sph-blackhole acc.
        """
        return self._set_acc(iobj, jobj)

    def set_phi_sph2bh(self, iobj, jobj):
        """
        Set sph-blackhole phi.
        """
        return self._set_phi(iobj, jobj)


class PostNewtonian(object):
    """
    A base class for post-newtonian gravity.
    """
    def __init__(self, clight, pnacc_kernel, pnphi_kernel):
        self._clight = clight
        self._pnacc_kernel = pnacc_kernel
        self._pnphi_kernel = pnphi_kernel


    # blackhole-blackhole
    def set_acc_bh2bh(self, iobj, jobj):
        """
        Set blackhole-blackhole pn-acc.
        """
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        clight = self._clight
        data = (np.uint32(ni),
                np.uint32(nj),
                iposmass, iobj.vel,
                jposmass, jobj.vel,
                np.uint32(clight.pn_order), np.float64(clight.inv1),
                np.float64(clight.inv2), np.float64(clight.inv3),
                np.float64(clight.inv4), np.float64(clight.inv5),
                np.float64(clight.inv6), np.float64(clight.inv7)
               )
        self._pnacc_kernel.load_data(*data)
        self._pnacc_kernel.execute()
        ret = self._pnacc_kernel.get_result()
        return (ret[:,:3], ret[:,3])


    def set_phi_bh2bh(self, iobj, jobj):
        """
        Set blackhole-blackhole pn-phi.
        """
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.vstack((iobj.pos.T, iobj.mass)).T
        jposmass = np.vstack((jobj.pos.T, jobj.mass)).T
        data = (np.uint32(ni),
                np.uint32(nj),
                iposmass, iobj.eps2,
                jposmass, jobj.eps2)

        output_buf = np.empty(ni)

        # Adjusts global_size to be an integer multiple of local_size
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size

        self._pnphi_kernel.load_data(*data, global_size=global_size,
                                     local_size=local_size,
                                     output_buf=output_buf)
        self._pnphi_kernel.execute()
        ret = self._pnphi_kernel.get_result()
        return ret


class Gravity(object):
    """
    A base class for gravitational interaction between different particle types.
    """
    def __init__(self, pn_order=4, clight=25.0):
        self._clight = Clight(pn_order, clight)
        self.newtonian = None
        self.post_newtonian = None
        self._has_built = False


    def build(self):
        if not self._has_built:
            kernels = extensions.build_kernels()
            clight = self._clight
            self.newtonian = Newtonian(kernels["cl_lib64_p2p_acc_kernel"],
                                       kernels["cl_lib64_p2p_phi_kernel"])
            self.post_newtonian = PostNewtonian(clight,
                                                kernels["c_lib64_p2p_pnacc_kernel"],
                                                kernels["cl_lib64_p2p_phi_kernel"])
            self._has_built = True


gravity_kernels = Gravity()


########## end of file ##########
