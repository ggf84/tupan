#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements a minimal class for gravitational interactions between
different particle types at Newtonian and post-Newtonian approach.
"""


import sys
import numpy as np
from collections import namedtuple
from pynbody.lib.utils import timings
from pynbody.lib.clkernels import (cl_p2p_acc, cl_p2p_phi)
try:
    from pynbody.lib.libgravity import (c_p2p_acc, c_p2p_phi, c_p2p_pnacc)
except:
    pass


__all__ = ['Gravity']


KERNELS = None
HAS_BUILT = False
Kernels = namedtuple("Kernels", ["set_acc", "set_phi", "set_pnacc"])




def cl_set_acc(bi, bj):
    ret = cl_p2p_acc.run(bi, bj)
    return (ret[:,:3], ret[:,3])

def cl_set_phi(bi, bj):
    phi = cl_p2p_phi.run(bi, bj)
    return phi

@timings
def c_set_acc(bi, bj):
    ret = c_p2p_acc(bi, bj)
    return (ret[:,:3], ret[:,3])

@timings
def c_set_phi(bi, bj):
    phi = c_p2p_phi(bi, bj)
    return phi

def c_set_pnacc(bi, bj, *optargs):
    ret = c_p2p_pnacc(bi, bj, optargs)
    return (ret[:,:3], ret[:,3])






class Clight(object):
    """

    """
    def __init__(self, clight):
        self._clight = clight
        self.inv1 = 1.0/clight
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
    def __init__(self):
        pass

    def setup_kernels(self, kernels):
        self._set_acc = kernels.set_acc
        self._set_phi = kernels.set_phi

    # body-body
    def set_acc_b2b(self, iobj, jobj):
        """
        Set body-body acc.
        """
        return self._set_acc(iobj, jobj)

#        ret = self._set_acc(iobj, jobj)

#        elapsed = self._set_acc.selftimer.elapsed
#        print('Execution time of C kernel: {0:g} s'.format(elapsed))
#        print('C kernel Gflops/s: {0:g}'.format(24*len(iobj)*len(jobj)/(elapsed*1e9)))

#        return ret


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
    def __init__(self, pn_order, clight):
        self._pn_order = pn_order
        self._clight = clight

    def setup_kernels(self, kernels):
        self._set_pnacc = kernels.set_pnacc
        self._set_pnphi = kernels.set_phi

    # blackhole-blackhole
    def set_acc_bh2bh(self, iobj, jobj):
        """
        Set blackhole-blackhole pn-acc.
        """
        return self._set_pnacc(iobj, jobj, self._pn_order,
                               self._clight.inv1, self._clight.inv2,
                               self._clight.inv3, self._clight.inv4,
                               self._clight.inv5, self._clight.inv6,
                               self._clight.inv7)

    def set_phi_bh2bh(self, iobj, jobj):
        """
        Set blackhole-blackhole pn-phi.
        """
        return self._set_pnphi(iobj, jobj)





class Gravity(object):
    """
    A base class for gravitational interaction between different particle types.
    """
    def __init__(self, pn_order=4, clight=25.0):
        self.newtonian = Newtonian()
        self.post_newtonian = PostNewtonian(pn_order, Clight(clight))


    def build_kernels(self):
        global HAS_BUILT
        if not HAS_BUILT:
            try:
#                raise
                if not cl_p2p_acc.has_built():
                    cl_p2p_acc.build_kernel()
                if not cl_p2p_phi.has_built():
                    cl_p2p_phi.build_kernel()
#                raise
            except Exception as e:
                cl_p2p_acc.is_available = False
                cl_p2p_phi.is_available = False
                print(e)
                ans = raw_input("A problem occurred with the loading of the OpenCL "
                                "kernels.\nAttempting to continue with C extensions "
                                "on the CPU only.\nDo you want to continue ([y]/n)? ")
                ans = ans.lower()
                if ans == 'n' or ans == 'no':
                    print('exiting...')
                    sys.exit(0)
            HAS_BUILT = True

        global KERNELS
        if not KERNELS:
            if cl_p2p_phi.is_available:
                set_phi = cl_set_phi
            else:
                set_phi = c_set_phi

            if cl_p2p_acc.is_available:
                set_acc = cl_set_acc
            else:
                set_acc = c_set_acc

            set_pnacc = c_set_pnacc

            KERNELS = Kernels(set_acc, set_phi, set_pnacc)

        kernels = KERNELS
        self.newtonian.setup_kernels(kernels)
        self.post_newtonian.setup_kernels(kernels)


########## end of file ##########
