# -*- coding: utf-8 -*-
#

"""
This module implements a minimal class for gravitational interactions
between particles in Newtonian and post-Newtonian approach.
"""


from __future__ import print_function, division
import sys
from .utils import ctype
from .extensions import get_kernel
from .utils.timing import decallmethods, timings


__all__ = ["Phi", "phi",
           "Acc", "acc",
           "AccJerk", "acc_jerk",
           "Tstep", "tstep",
           "PNAcc", "pnacc",
           "Sakura", "sakura",
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
        self.kernel.local_size = 512
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz)

        self.kernel.set_args(*args)

        self.osize = ni
        if ni > self.max_output_size:
            self.kernel.set_local_memory(18, 1)
            self.kernel.set_local_memory(19, 1)
            self.kernel.set_local_memory(20, 1)
            self.kernel.set_local_memory(21, 1)
            self.kernel.set_local_memory(22, 1)
            self.kernel.set_local_memory(23, 1)
            self.kernel.set_local_memory(24, 1)
            self.kernel.set_local_memory(25, 1)
            self.phi = self.kernel.allocate_buffer(26, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(26, self.phi)
        return self.phi[:ni]


@decallmethods(timings)
class Acc(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("acc_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz)

        self.kernel.set_args(*args)

        self.osize = ni
        if ni > self.max_output_size:
            self.kernel.set_local_memory(18, 1)
            self.kernel.set_local_memory(19, 1)
            self.kernel.set_local_memory(20, 1)
            self.kernel.set_local_memory(21, 1)
            self.kernel.set_local_memory(22, 1)
            self.kernel.set_local_memory(23, 1)
            self.kernel.set_local_memory(24, 1)
            self.kernel.set_local_memory(25, 1)
            self.ax = self.kernel.allocate_buffer(26, ni, ctype.REAL)
            self.ay = self.kernel.allocate_buffer(27, ni, ctype.REAL)
            self.az = self.kernel.allocate_buffer(28, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(26, self.ax)
        self.kernel.map_buffer(27, self.ay)
        self.kernel.map_buffer(28, self.az)
        return [self.ax[:ni], self.ay[:ni], self.az[:ni]]


@decallmethods(timings)
class AccJerk(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("acc_jerk_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz)

        self.kernel.set_args(*args)

        self.osize = ni
        if ni > self.max_output_size:
            self.kernel.set_local_memory(18, 1)
            self.kernel.set_local_memory(19, 1)
            self.kernel.set_local_memory(20, 1)
            self.kernel.set_local_memory(21, 1)
            self.kernel.set_local_memory(22, 1)
            self.kernel.set_local_memory(23, 1)
            self.kernel.set_local_memory(24, 1)
            self.kernel.set_local_memory(25, 1)
            self.ax = self.kernel.allocate_buffer(26, ni, ctype.REAL)
            self.ay = self.kernel.allocate_buffer(27, ni, ctype.REAL)
            self.az = self.kernel.allocate_buffer(28, ni, ctype.REAL)
            self.jx = self.kernel.allocate_buffer(29, ni, ctype.REAL)
            self.jy = self.kernel.allocate_buffer(30, ni, ctype.REAL)
            self.jz = self.kernel.allocate_buffer(31, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(26, self.ax)
        self.kernel.map_buffer(27, self.ay)
        self.kernel.map_buffer(28, self.az)
        self.kernel.map_buffer(29, self.jx)
        self.kernel.map_buffer(30, self.jy)
        self.kernel.map_buffer(31, self.jz)
        return [self.ax[:ni], self.ay[:ni], self.az[:ni],
                self.jx[:ni], self.jy[:ni], self.jz[:ni]]


@decallmethods(timings)
class Tstep(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("tstep_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

    def set_args(self, iobj, jobj, eta):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                eta)

        self.kernel.set_args(*args)

        self.osize = ni
        if ni > self.max_output_size:
            self.kernel.set_local_memory(19, 1)
            self.kernel.set_local_memory(20, 1)
            self.kernel.set_local_memory(21, 1)
            self.kernel.set_local_memory(22, 1)
            self.kernel.set_local_memory(23, 1)
            self.kernel.set_local_memory(24, 1)
            self.kernel.set_local_memory(25, 1)
            self.kernel.set_local_memory(26, 1)
            self.tstep_a = self.kernel.allocate_buffer(27, ni, ctype.REAL)
            self.tstep_b = self.kernel.allocate_buffer(28, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(27, self.tstep_a)
        self.kernel.map_buffer(28, self.tstep_b)

        return (self.tstep_a[:ni], self.tstep_b[:ni])


@decallmethods(timings)
class PNAcc(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("pnacc_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                clight.pn_order, clight.inv1,
                clight.inv2, clight.inv3,
                clight.inv4, clight.inv5,
                clight.inv6, clight.inv7)

        self.kernel.set_args(*args)

        self.osize = ni
        if ni > self.max_output_size:
            self.kernel.set_local_memory(26, 1)
            self.kernel.set_local_memory(27, 1)
            self.kernel.set_local_memory(28, 1)
            self.kernel.set_local_memory(29, 1)
            self.kernel.set_local_memory(30, 1)
            self.kernel.set_local_memory(31, 1)
            self.kernel.set_local_memory(32, 1)
            self.kernel.set_local_memory(33, 1)
            self.pnax = self.kernel.allocate_buffer(34, ni, ctype.REAL)
            self.pnay = self.kernel.allocate_buffer(35, ni, ctype.REAL)
            self.pnaz = self.kernel.allocate_buffer(36, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(34, self.pnax)
        self.kernel.map_buffer(35, self.pnay)
        self.kernel.map_buffer(36, self.pnaz)
        return [self.pnax[:ni], self.pnay[:ni], self.pnaz[:ni]]


@decallmethods(timings)
class Sakura(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("sakura_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                dt)

        self.kernel.set_args(*args)

        self.osize = ni
        if ni > self.max_output_size:
            self.kernel.set_local_memory(19, 1)
            self.kernel.set_local_memory(20, 1)
            self.kernel.set_local_memory(21, 1)
            self.kernel.set_local_memory(22, 1)
            self.kernel.set_local_memory(23, 1)
            self.kernel.set_local_memory(24, 1)
            self.kernel.set_local_memory(25, 1)
            self.kernel.set_local_memory(26, 1)
            self.drx = self.kernel.allocate_buffer(27, ni, ctype.REAL)
            self.dry = self.kernel.allocate_buffer(28, ni, ctype.REAL)
            self.drz = self.kernel.allocate_buffer(29, ni, ctype.REAL)
            self.dvx = self.kernel.allocate_buffer(30, ni, ctype.REAL)
            self.dvy = self.kernel.allocate_buffer(31, ni, ctype.REAL)
            self.dvz = self.kernel.allocate_buffer(32, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(27, self.drx)
        self.kernel.map_buffer(28, self.dry)
        self.kernel.map_buffer(29, self.drz)
        self.kernel.map_buffer(30, self.dvx)
        self.kernel.map_buffer(31, self.dvy)
        self.kernel.map_buffer(32, self.dvz)
        return [self.drx[:ni], self.dry[:ni], self.drz[:ni],
                self.dvx[:ni], self.dvy[:ni], self.dvz[:ni]]


@decallmethods(timings)
class NREG_X(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("nreg_Xkernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni

        args = (ni,
                iobj.mass, iobj.x, iobj.y, iobj.z,
                iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                nj,
                jobj.mass, jobj.x, jobj.y, jobj.z,
                jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                dt)

        self.kernel.set_args(*args)

        self.osize = ni
        if ni > self.max_output_size:
            self.kernel.set_local_memory(19, 1)
            self.kernel.set_local_memory(20, 1)
            self.kernel.set_local_memory(21, 1)
            self.kernel.set_local_memory(22, 1)
            self.kernel.set_local_memory(23, 1)
            self.kernel.set_local_memory(24, 1)
            self.kernel.set_local_memory(25, 1)
            self.kernel.set_local_memory(26, 1)
            self.rx = self.kernel.allocate_buffer(27, ni, ctype.REAL)
            self.ry = self.kernel.allocate_buffer(28, ni, ctype.REAL)
            self.rz = self.kernel.allocate_buffer(29, ni, ctype.REAL)
            self.ax = self.kernel.allocate_buffer(30, ni, ctype.REAL)
            self.ay = self.kernel.allocate_buffer(31, ni, ctype.REAL)
            self.az = self.kernel.allocate_buffer(32, ni, ctype.REAL)
            self.u = self.kernel.allocate_buffer(33, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(27, self.rx)
        self.kernel.map_buffer(28, self.ry)
        self.kernel.map_buffer(29, self.rz)
        self.kernel.map_buffer(30, self.ax)
        self.kernel.map_buffer(31, self.ay)
        self.kernel.map_buffer(32, self.az)
        self.kernel.map_buffer(33, self.u)
        return [self.rx[:ni], self.ry[:ni], self.rz[:ni],
                self.ax[:ni], self.ay[:ni], self.az[:ni],
                0.5 * self.u[:ni].sum()]


@decallmethods(timings)
class NREG_V(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("nreg_Vkernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni

        args = (ni,
                iobj.mass, iobj.vx, iobj.vy, iobj.vz,
                iobj.ax, iobj.ay, iobj.az,
                nj,
                jobj.mass, jobj.vx, jobj.vy, jobj.vz,
                jobj.ax, jobj.ay, jobj.az,
                dt)

        self.kernel.set_args(*args)

        self.osize = ni
        if ni > self.max_output_size:
            self.kernel.set_local_memory(17, 1)
            self.kernel.set_local_memory(18, 1)
            self.kernel.set_local_memory(19, 1)
            self.kernel.set_local_memory(20, 1)
            self.kernel.set_local_memory(21, 1)
            self.kernel.set_local_memory(22, 1)
            self.kernel.set_local_memory(23, 1)
            self.vx = self.kernel.allocate_buffer(24, ni, ctype.REAL)
            self.vy = self.kernel.allocate_buffer(25, ni, ctype.REAL)
            self.vz = self.kernel.allocate_buffer(26, ni, ctype.REAL)
            self.k = self.kernel.allocate_buffer(27, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(24, self.vx)
        self.kernel.map_buffer(25, self.vy)
        self.kernel.map_buffer(26, self.vz)
        self.kernel.map_buffer(27, self.k)
        return [self.vx[:ni], self.vy[:ni], self.vz[:ni],
                0.5 * self.k[:ni].sum()]


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
