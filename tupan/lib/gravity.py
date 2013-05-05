# -*- coding: utf-8 -*-
#

"""
This module implements a minimal class for gravitational interactions
between particles in Newtonian and post-Newtonian approach.
"""


from __future__ import print_function, division
import sys
import numpy as np
from .utils import ctype
from .extensions import get_kernel
from .utils.timing import decallmethods, timings


__all__ = ["Phi", "phi",
           "Acc", "acc",
           "AccJerk", "acc_jerk",
           "Tstep", "tstep",
           "PNAcc", "pnacc",
           "Sakura", "sakura",
           "NREG_X", "nreg_x",
           "NREG_V", "nreg_v",
           ]


@decallmethods(timings)
class Clight(object):
    """This class holds the values of the PN-order, the speed of light and
    some of its inverse powers.
    """
    def __init__(self):
        self._pn_order = 0
        self._clight = None

    @property
    def pn_order(self):
        return self._pn_order

    @pn_order.setter
    def pn_order(self, value):
        self._pn_order = int(value)

    @property
    def clight(self):
        return self._clight

    @clight.setter
    def clight(self, value):
        self._clight = float(value)
        self.inv1 = 1.0/self._clight
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
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("phi_kernel", exttype, prec)
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self.__jm = self.kernel.alloc_local_memory(512)
            self.__jrx = self.kernel.alloc_local_memory(512)
            self.__jry = self.kernel.alloc_local_memory(512)
            self.__jrz = self.kernel.alloc_local_memory(512)
            self.__je2 = self.kernel.alloc_local_memory(512)
            self.__jvx = self.kernel.alloc_local_memory(512)
            self.__jvy = self.kernel.alloc_local_memory(512)
            self.__jvz = self.kernel.alloc_local_memory(512)
            self._phi = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                self.__jm, self.__jrx, self.__jry, self.__jrz,
                self.__je2, self.__jvx, self.__jvy, self.__jvz,
                self._phi)

        self.kernel.set_args(*args)

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(26, self._phi)
        return self._phi[:ni]


@decallmethods(timings)
class Acc(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("acc_kernel", exttype, prec)
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self.__jm = self.kernel.alloc_local_memory(512)
            self.__jrx = self.kernel.alloc_local_memory(512)
            self.__jry = self.kernel.alloc_local_memory(512)
            self.__jrz = self.kernel.alloc_local_memory(512)
            self.__je2 = self.kernel.alloc_local_memory(512)
            self.__jvx = self.kernel.alloc_local_memory(512)
            self.__jvy = self.kernel.alloc_local_memory(512)
            self.__jvz = self.kernel.alloc_local_memory(512)
            self._ax = np.zeros(ni, dtype=ctype.REAL)
            self._ay = np.zeros(ni, dtype=ctype.REAL)
            self._az = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                self.__jm, self.__jrx, self.__jry, self.__jrz,
                self.__je2, self.__jvx, self.__jvy, self.__jvz,
                self._ax, self._ay, self._az)

        self.kernel.set_args(*args)

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(26, self._ax)
        self.kernel.map_buffer(27, self._ay)
        self.kernel.map_buffer(28, self._az)
        return [self._ax[:ni], self._ay[:ni], self._az[:ni]]


@decallmethods(timings)
class AccJerk(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("acc_jerk_kernel", exttype, prec)
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self.__jm = self.kernel.alloc_local_memory(512)
            self.__jrx = self.kernel.alloc_local_memory(512)
            self.__jry = self.kernel.alloc_local_memory(512)
            self.__jrz = self.kernel.alloc_local_memory(512)
            self.__je2 = self.kernel.alloc_local_memory(512)
            self.__jvx = self.kernel.alloc_local_memory(512)
            self.__jvy = self.kernel.alloc_local_memory(512)
            self.__jvz = self.kernel.alloc_local_memory(512)
            self._ax = np.zeros(ni, dtype=ctype.REAL)
            self._ay = np.zeros(ni, dtype=ctype.REAL)
            self._az = np.zeros(ni, dtype=ctype.REAL)
            self._jx = np.zeros(ni, dtype=ctype.REAL)
            self._jy = np.zeros(ni, dtype=ctype.REAL)
            self._jz = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                self.__jm, self.__jrx, self.__jry, self.__jrz,
                self.__je2, self.__jvx, self.__jvy, self.__jvz,
                self._ax, self._ay, self._az,
                self._jx, self._jy, self._jz)

        self.kernel.set_args(*args)

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(26, self._ax)
        self.kernel.map_buffer(27, self._ay)
        self.kernel.map_buffer(28, self._az)
        self.kernel.map_buffer(29, self._jx)
        self.kernel.map_buffer(30, self._jy)
        self.kernel.map_buffer(31, self._jz)
        return [self._ax[:ni], self._ay[:ni], self._az[:ni],
                self._jx[:ni], self._jy[:ni], self._jz[:ni]]


@decallmethods(timings)
class Tstep(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("tstep_kernel", exttype, prec)
        self.max_output_size = 0

    def set_args(self, iobj, jobj, eta):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self.__jm = self.kernel.alloc_local_memory(512)
            self.__jrx = self.kernel.alloc_local_memory(512)
            self.__jry = self.kernel.alloc_local_memory(512)
            self.__jrz = self.kernel.alloc_local_memory(512)
            self.__je2 = self.kernel.alloc_local_memory(512)
            self.__jvx = self.kernel.alloc_local_memory(512)
            self.__jvy = self.kernel.alloc_local_memory(512)
            self.__jvz = self.kernel.alloc_local_memory(512)
            self._tstep_a = np.zeros(ni, dtype=ctype.REAL)
            self._tstep_b = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                eta,
                self.__jm, self.__jrx, self.__jry, self.__jrz,
                self.__je2, self.__jvx, self.__jvy, self.__jvz,
                self._tstep_a, self._tstep_b)

        self.kernel.set_args(*args)

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(27, self._tstep_a)
        self.kernel.map_buffer(28, self._tstep_b)

        return (self._tstep_a[:ni], self._tstep_b[:ni])


@decallmethods(timings)
class PNAcc(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("pnacc_kernel", exttype, prec)
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self.__jm = self.kernel.alloc_local_memory(512)
            self.__jrx = self.kernel.alloc_local_memory(512)
            self.__jry = self.kernel.alloc_local_memory(512)
            self.__jrz = self.kernel.alloc_local_memory(512)
            self.__je2 = self.kernel.alloc_local_memory(512)
            self.__jvx = self.kernel.alloc_local_memory(512)
            self.__jvy = self.kernel.alloc_local_memory(512)
            self.__jvz = self.kernel.alloc_local_memory(512)
            self._pnax = np.zeros(ni, dtype=ctype.REAL)
            self._pnay = np.zeros(ni, dtype=ctype.REAL)
            self._pnaz = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                clight.pn_order, clight.inv1,
                clight.inv2, clight.inv3,
                clight.inv4, clight.inv5,
                clight.inv6, clight.inv7,
                self.__jm, self.__jrx, self.__jry, self.__jrz,
                self.__je2, self.__jvx, self.__jvy, self.__jvz,
                self._pnax, self._pnay, self._pnaz)

        self.kernel.set_args(*args)

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(34, self._pnax)
        self.kernel.map_buffer(35, self._pnay)
        self.kernel.map_buffer(36, self._pnaz)
        return [self._pnax[:ni], self._pnay[:ni], self._pnaz[:ni]]


@decallmethods(timings)
class Sakura(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("sakura_kernel", exttype, prec)
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self.__jm = self.kernel.alloc_local_memory(512)
            self.__jrx = self.kernel.alloc_local_memory(512)
            self.__jry = self.kernel.alloc_local_memory(512)
            self.__jrz = self.kernel.alloc_local_memory(512)
            self.__je2 = self.kernel.alloc_local_memory(512)
            self.__jvx = self.kernel.alloc_local_memory(512)
            self.__jvy = self.kernel.alloc_local_memory(512)
            self.__jvz = self.kernel.alloc_local_memory(512)
            self._drx = np.zeros(ni, dtype=ctype.REAL)
            self._dry = np.zeros(ni, dtype=ctype.REAL)
            self._drz = np.zeros(ni, dtype=ctype.REAL)
            self._dvx = np.zeros(ni, dtype=ctype.REAL)
            self._dvy = np.zeros(ni, dtype=ctype.REAL)
            self._dvz = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                dt,
                self.__jm, self.__jrx, self.__jry, self.__jrz,
                self.__je2, self.__jvx, self.__jvy, self.__jvz,
                self._drx, self._dry, self._drz,
                self._dvx, self._dvy, self._dvz)

        self.kernel.set_args(*args)

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(27, self._drx)
        self.kernel.map_buffer(28, self._dry)
        self.kernel.map_buffer(29, self._drz)
        self.kernel.map_buffer(30, self._dvx)
        self.kernel.map_buffer(31, self._dvy)
        self.kernel.map_buffer(32, self._dvz)
        return [self._drx[:ni], self._dry[:ni], self._drz[:ni],
                self._dvx[:ni], self._dvy[:ni], self._dvz[:ni]]


@decallmethods(timings)
class NREG_X(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("nreg_Xkernel", exttype, prec)
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self.__jm = self.kernel.alloc_local_memory(512)
            self.__jrx = self.kernel.alloc_local_memory(512)
            self.__jry = self.kernel.alloc_local_memory(512)
            self.__jrz = self.kernel.alloc_local_memory(512)
            self.__je2 = self.kernel.alloc_local_memory(512)
            self.__jvx = self.kernel.alloc_local_memory(512)
            self.__jvy = self.kernel.alloc_local_memory(512)
            self.__jvz = self.kernel.alloc_local_memory(512)
            self._rx = np.zeros(ni, dtype=ctype.REAL)
            self._ry = np.zeros(ni, dtype=ctype.REAL)
            self._rz = np.zeros(ni, dtype=ctype.REAL)
            self._ax = np.zeros(ni, dtype=ctype.REAL)
            self._ay = np.zeros(ni, dtype=ctype.REAL)
            self._az = np.zeros(ni, dtype=ctype.REAL)
            self._u = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                dt,
                self.__jm, self.__jrx, self.__jry, self.__jrz,
                self.__je2, self.__jvx, self.__jvy, self.__jvz,
                self._rx, self._ry, self._rz,
                self._ax, self._ay, self._az,
                self._u)

        self.kernel.set_args(*args)

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(27, self._rx)
        self.kernel.map_buffer(28, self._ry)
        self.kernel.map_buffer(29, self._rz)
        self.kernel.map_buffer(30, self._ax)
        self.kernel.map_buffer(31, self._ay)
        self.kernel.map_buffer(32, self._az)
        self.kernel.map_buffer(33, self._u)
        return [self._rx[:ni], self._ry[:ni], self._rz[:ni],
                self._ax[:ni], self._ay[:ni], self._az[:ni],
                0.5 * self._u[:ni].sum()]


@decallmethods(timings)
class NREG_V(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("nreg_Vkernel", exttype, prec)
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self.__jm = self.kernel.alloc_local_memory(512)
            self.__jvx = self.kernel.alloc_local_memory(512)
            self.__jvy = self.kernel.alloc_local_memory(512)
            self.__jvz = self.kernel.alloc_local_memory(512)
            self.__jax = self.kernel.alloc_local_memory(512)
            self.__jay = self.kernel.alloc_local_memory(512)
            self.__jaz = self.kernel.alloc_local_memory(512)
            self._vx = np.zeros(ni, dtype=ctype.REAL)
            self._vy = np.zeros(ni, dtype=ctype.REAL)
            self._vz = np.zeros(ni, dtype=ctype.REAL)
            self._k = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        args = (ni,
                iobj.mass, iobj.vx, iobj.vy, iobj.vz,
                iobj.ax, iobj.ay, iobj.az,
                nj,
                jobj.mass, jobj.vx, jobj.vy, jobj.vz,
                jobj.ax, jobj.ay, jobj.az,
                dt,
                self.__jm, self.__jvx, self.__jvy, self.__jvz,
                self.__jax, self.__jay, self.__jaz,
                self._vx, self._vy, self._vz,
                self._k)

        self.kernel.set_args(*args)

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(24, self._vx)
        self.kernel.map_buffer(25, self._vy)
        self.kernel.map_buffer(26, self._vz)
        self.kernel.map_buffer(27, self._k)
        return [self._vx[:ni], self._vy[:ni], self._vz[:ni],
                0.5 * self._k[:ni].sum()]


exttype = "cl" if "--use_cl" in sys.argv else "c"

clight = Clight()
phi = Phi(exttype, ctype.prec)
acc = Acc(exttype, ctype.prec)
acc_jerk = AccJerk(exttype, ctype.prec)
tstep = Tstep(exttype, ctype.prec)
pnacc = PNAcc(exttype, ctype.prec)
sakura = Sakura(exttype, ctype.prec)
nreg_x = NREG_X(exttype, ctype.prec)
nreg_v = NREG_V(exttype, ctype.prec)


########## end of file ##########
