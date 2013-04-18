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

        self.kernel.set_local_memory(19, 1)
        self.kernel.set_local_memory(20, 1)
        self.kernel.set_local_memory(21, 1)
        self.kernel.set_local_memory(22, 1)
        self.kernel.set_local_memory(23, 1)
        self.kernel.set_local_memory(24, 1)
        self.kernel.set_local_memory(25, 1)
        self.kernel.set_local_memory(26, 1)

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_array(1, iobj.x)
        self.kernel.set_array(2, iobj.y)
        self.kernel.set_array(3, iobj.z)
        self.kernel.set_array(4, iobj.mass)
        self.kernel.set_array(5, iobj.vx)
        self.kernel.set_array(6, iobj.vy)
        self.kernel.set_array(7, iobj.vz)
        self.kernel.set_array(8, iobj.eps2)
        self.kernel.set_int(9, nj)
        self.kernel.set_array(10, jobj.x)
        self.kernel.set_array(11, jobj.y)
        self.kernel.set_array(12, jobj.z)
        self.kernel.set_array(13, jobj.mass)
        self.kernel.set_array(14, jobj.vx)
        self.kernel.set_array(15, jobj.vy)
        self.kernel.set_array(16, jobj.vz)
        self.kernel.set_array(17, jobj.eps2)

        self.osize = ni
        if ni > self.max_output_size:
            self.phi = self.kernel.allocate_buffer(18, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(18, self.phi)
        return self.phi[:ni]


@decallmethods(timings)
class Acc(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("acc_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

        self.kernel.set_local_memory(21, 1)
        self.kernel.set_local_memory(22, 1)
        self.kernel.set_local_memory(23, 1)
        self.kernel.set_local_memory(24, 1)
        self.kernel.set_local_memory(25, 1)
        self.kernel.set_local_memory(26, 1)
        self.kernel.set_local_memory(27, 1)
        self.kernel.set_local_memory(28, 1)

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_array(1, iobj.x)
        self.kernel.set_array(2, iobj.y)
        self.kernel.set_array(3, iobj.z)
        self.kernel.set_array(4, iobj.mass)
        self.kernel.set_array(5, iobj.vx)
        self.kernel.set_array(6, iobj.vy)
        self.kernel.set_array(7, iobj.vz)
        self.kernel.set_array(8, iobj.eps2)
        self.kernel.set_int(9, nj)
        self.kernel.set_array(10, jobj.x)
        self.kernel.set_array(11, jobj.y)
        self.kernel.set_array(12, jobj.z)
        self.kernel.set_array(13, jobj.mass)
        self.kernel.set_array(14, jobj.vx)
        self.kernel.set_array(15, jobj.vy)
        self.kernel.set_array(16, jobj.vz)
        self.kernel.set_array(17, jobj.eps2)

        self.osize = ni
        if ni > self.max_output_size:
            self.ax = self.kernel.allocate_buffer(18, ni, ctype.REAL)
            self.ay = self.kernel.allocate_buffer(19, ni, ctype.REAL)
            self.az = self.kernel.allocate_buffer(20, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(18, self.ax)
        self.kernel.map_buffer(19, self.ay)
        self.kernel.map_buffer(20, self.az)
        return [self.ax[:ni], self.ay[:ni], self.az[:ni]]


@decallmethods(timings)
class AccJerk(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("acc_jerk_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

        self.kernel.set_local_memory(24, 1)
        self.kernel.set_local_memory(25, 1)
        self.kernel.set_local_memory(26, 1)
        self.kernel.set_local_memory(27, 1)
        self.kernel.set_local_memory(28, 1)
        self.kernel.set_local_memory(29, 1)
        self.kernel.set_local_memory(30, 1)
        self.kernel.set_local_memory(31, 1)

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_array(1, iobj.x)
        self.kernel.set_array(2, iobj.y)
        self.kernel.set_array(3, iobj.z)
        self.kernel.set_array(4, iobj.mass)
        self.kernel.set_array(5, iobj.vx)
        self.kernel.set_array(6, iobj.vy)
        self.kernel.set_array(7, iobj.vz)
        self.kernel.set_array(8, iobj.eps2)
        self.kernel.set_int(9, nj)
        self.kernel.set_array(10, jobj.x)
        self.kernel.set_array(11, jobj.y)
        self.kernel.set_array(12, jobj.z)
        self.kernel.set_array(13, jobj.mass)
        self.kernel.set_array(14, jobj.vx)
        self.kernel.set_array(15, jobj.vy)
        self.kernel.set_array(16, jobj.vz)
        self.kernel.set_array(17, jobj.eps2)

        self.osize = ni
        if ni > self.max_output_size:
            self.ax = self.kernel.allocate_buffer(18, ni, ctype.REAL)
            self.ay = self.kernel.allocate_buffer(19, ni, ctype.REAL)
            self.az = self.kernel.allocate_buffer(20, ni, ctype.REAL)
            self.jx = self.kernel.allocate_buffer(21, ni, ctype.REAL)
            self.jy = self.kernel.allocate_buffer(22, ni, ctype.REAL)
            self.jz = self.kernel.allocate_buffer(23, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(18, self.ax)
        self.kernel.map_buffer(19, self.ay)
        self.kernel.map_buffer(20, self.az)
        self.kernel.map_buffer(21, self.jx)
        self.kernel.map_buffer(22, self.jy)
        self.kernel.map_buffer(23, self.jz)
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
#        self._count = 0

        self.kernel.set_local_memory(20, 1)
        self.kernel.set_local_memory(21, 1)
        self.kernel.set_local_memory(22, 1)
        self.kernel.set_local_memory(23, 1)
        self.kernel.set_local_memory(24, 1)
        self.kernel.set_local_memory(25, 1)
        self.kernel.set_local_memory(26, 1)
        self.kernel.set_local_memory(27, 1)

    def set_args(self, iobj, jobj, eta):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_array(1, iobj.x)
        self.kernel.set_array(2, iobj.y)
        self.kernel.set_array(3, iobj.z)
        self.kernel.set_array(4, iobj.mass)
        self.kernel.set_array(5, iobj.vx)
        self.kernel.set_array(6, iobj.vy)
        self.kernel.set_array(7, iobj.vz)
        self.kernel.set_array(8, iobj.eps2)
        self.kernel.set_int(9, nj)
        self.kernel.set_array(10, jobj.x)
        self.kernel.set_array(11, jobj.y)
        self.kernel.set_array(12, jobj.z)
        self.kernel.set_array(13, jobj.mass)
        self.kernel.set_array(14, jobj.vx)
        self.kernel.set_array(15, jobj.vy)
        self.kernel.set_array(16, jobj.vz)
        self.kernel.set_array(17, jobj.eps2)
        self.kernel.set_float(18, eta)

        self.osize = ni
        if ni > self.max_output_size:
            self.tstep = self.kernel.allocate_buffer(19, ni, ctype.REAL)
            self.ijtstepmin = self.kernel.allocate_buffer(20, 3, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(19, self.tstep)
        self.kernel.map_buffer(20, self.ijtstepmin)

#        i = int(self.ijtstepmin[0])
#        j = int(self.ijtstepmin[1])
#        ijstepmin = self.ijtstepmin[2]
#
#        iw2 = 1/self.tstep[:ni]**2
#        ijw2max = 1/ijstepmin**2
#        w2_sakura = iw2[i] + iw2[j] - 2*ijw2max
#
#        with open("dts0.dat", "a") as fobj:
#            print(self._count, i, j,
#                1/ijw2max**0.5,
#                1/w2_sakura**0.5,
#                (1/(iw2**0.5)).min(),
#                file=fobj)
#            self._count += 1

        return self.tstep[:ni]
#        return (i, j, ijstepmin, self.tstep[:ni])


@decallmethods(timings)
class PNAcc(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("pnacc_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

        self.kernel.set_local_memory(29, 1)
        self.kernel.set_local_memory(30, 1)
        self.kernel.set_local_memory(31, 1)
        self.kernel.set_local_memory(32, 1)
        self.kernel.set_local_memory(33, 1)
        self.kernel.set_local_memory(34, 1)
        self.kernel.set_local_memory(35, 1)
        self.kernel.set_local_memory(36, 1)

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_array(1, iobj.x)
        self.kernel.set_array(2, iobj.y)
        self.kernel.set_array(3, iobj.z)
        self.kernel.set_array(4, iobj.mass)
        self.kernel.set_array(5, iobj.vx)
        self.kernel.set_array(6, iobj.vy)
        self.kernel.set_array(7, iobj.vz)
        self.kernel.set_array(8, iobj.eps2)
        self.kernel.set_int(9, nj)
        self.kernel.set_array(10, jobj.x)
        self.kernel.set_array(11, jobj.y)
        self.kernel.set_array(12, jobj.z)
        self.kernel.set_array(13, jobj.mass)
        self.kernel.set_array(14, jobj.vx)
        self.kernel.set_array(15, jobj.vy)
        self.kernel.set_array(16, jobj.vz)
        self.kernel.set_array(17, jobj.eps2)
        self.kernel.set_int(18, clight.pn_order)
        self.kernel.set_float(19, clight.inv1)
        self.kernel.set_float(20, clight.inv2)
        self.kernel.set_float(21, clight.inv3)
        self.kernel.set_float(22, clight.inv4)
        self.kernel.set_float(23, clight.inv5)
        self.kernel.set_float(24, clight.inv6)
        self.kernel.set_float(25, clight.inv7)

        self.osize = ni
        if ni > self.max_output_size:
            self.pnax = self.kernel.allocate_buffer(26, ni, ctype.REAL)
            self.pnay = self.kernel.allocate_buffer(27, ni, ctype.REAL)
            self.pnaz = self.kernel.allocate_buffer(28, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(26, self.pnax)
        self.kernel.map_buffer(27, self.pnay)
        self.kernel.map_buffer(28, self.pnaz)
        return [self.pnax[:ni], self.pnay[:ni], self.pnaz[:ni]]


@decallmethods(timings)
class Sakura(object):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("sakura_kernel", exttype, prec)
        self.kernel.local_size = 512
        self.max_output_size = 0

        self.kernel.set_local_memory(25, 1)
        self.kernel.set_local_memory(26, 1)
        self.kernel.set_local_memory(27, 1)
        self.kernel.set_local_memory(28, 1)
        self.kernel.set_local_memory(29, 1)
        self.kernel.set_local_memory(30, 1)
        self.kernel.set_local_memory(31, 1)
        self.kernel.set_local_memory(32, 1)

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_array(1, iobj.x)
        self.kernel.set_array(2, iobj.y)
        self.kernel.set_array(3, iobj.z)
        self.kernel.set_array(4, iobj.mass)
        self.kernel.set_array(5, iobj.vx)
        self.kernel.set_array(6, iobj.vy)
        self.kernel.set_array(7, iobj.vz)
        self.kernel.set_array(8, iobj.eps2)
        self.kernel.set_int(9, nj)
        self.kernel.set_array(10, jobj.x)
        self.kernel.set_array(11, jobj.y)
        self.kernel.set_array(12, jobj.z)
        self.kernel.set_array(13, jobj.mass)
        self.kernel.set_array(14, jobj.vx)
        self.kernel.set_array(15, jobj.vy)
        self.kernel.set_array(16, jobj.vz)
        self.kernel.set_array(17, jobj.eps2)
        self.kernel.set_float(18, dt)

        self.osize = ni
        if ni > self.max_output_size:
            self.drx = self.kernel.allocate_buffer(19, ni, ctype.REAL)
            self.dry = self.kernel.allocate_buffer(20, ni, ctype.REAL)
            self.drz = self.kernel.allocate_buffer(21, ni, ctype.REAL)
            self.dvx = self.kernel.allocate_buffer(22, ni, ctype.REAL)
            self.dvy = self.kernel.allocate_buffer(23, ni, ctype.REAL)
            self.dvz = self.kernel.allocate_buffer(24, ni, ctype.REAL)
            self.max_output_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        ni = self.osize
        self.kernel.map_buffer(19, self.drx)
        self.kernel.map_buffer(20, self.dry)
        self.kernel.map_buffer(21, self.drz)
        self.kernel.map_buffer(22, self.dvx)
        self.kernel.map_buffer(23, self.dvy)
        self.kernel.map_buffer(24, self.dvz)
        return [self.drx[:ni], self.dry[:ni], self.drz[:ni],
                self.dvx[:ni], self.dvy[:ni], self.dvz[:ni]]


exttype = "cl" if "--use_cl" in sys.argv else "c"

clight = Clight()
phi = Phi(exttype, ctype.prec)
acc = Acc(exttype, ctype.prec)
acc_jerk = AccJerk(exttype, ctype.prec)
tstep = Tstep(exttype, ctype.prec)
pnacc = PNAcc(exttype, ctype.prec)
sakura = Sakura(exttype, ctype.prec)


########## end of file ##########
