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


def prepare_args(args, argtypes):
    return [argtype(arg) for (arg, argtype) in zip(args, argtypes)]


class AbstractExtension(object):
    def set_args(self, iobj, jobj, *args):
        raise NotImplemented

    def run(self):
        self.kernel.run()

    def get_result(self):
        return self.kernel.map_buffers(self._outargs, self.outargs)

    def calc(self, iobj, jobj, *args):
        self.set_args(iobj, jobj, *args)
        self.run()
        return self.get_result()


@decallmethods(timings)
class Phi(AbstractExtension):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("phi_kernel", exttype, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p)
        restypes = (cty.c_real_p,)
        self.argtypes = argtypes
        self.restypes = restypes
        lmem = self.kernel.allocate_local_memory(8, np.dtype(ctype.REAL))
        self.kernel.set_args(lmem, start=19)
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self._phi = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz)
        self._outargs = (self._phi[:ni],)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)

    def _pycalc(self, iobj, jobj):
        # Never use this method for production runs. It is very slow
        # and is here only for performance comparisons. It is also
        # likely that only the classes Acc and Phi will have an
        # implementation of this method.
        ni = iobj.n
        if ni > self.max_output_size:
            self._phi = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni
        for i in range(ni):
            rx = iobj.rx[i] - jobj.rx
            ry = iobj.ry[i] - jobj.ry
            rz = iobj.rz[i] - jobj.rz
            e2 = iobj.eps2[i] + jobj.eps2
            r2 = rx * rx + ry * ry + rz * rz
            mask = r2 > 0
            inv_r2 = 1 / (r2 + e2)
            inv_r = np.sqrt(inv_r2)
            self._phi[i] = -(jobj.mass * inv_r)[mask].sum()
        return (self._phi[:ni],)
#    calc = _pycalc


@decallmethods(timings)
class Acc(AbstractExtension):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("acc_kernel", exttype, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes
        lmem = self.kernel.allocate_local_memory(8, np.dtype(ctype.REAL))
        self.kernel.set_args(lmem, start=21)
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self._ax = np.zeros(ni, dtype=ctype.REAL)
            self._ay = np.zeros(ni, dtype=ctype.REAL)
            self._az = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz)
        self._outargs = (self._ax[:ni], self._ay[:ni], self._az[:ni])

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)

    def _pycalc(self, iobj, jobj):
        # Never use this method for production runs. It is very slow
        # and is here only for performance comparisons. It is also
        # likely that only the classes Acc and Phi will have an
        # implementation of this method.
        ni = iobj.n
        if ni > self.max_output_size:
            self._ax = np.zeros(ni, dtype=ctype.REAL)
            self._ay = np.zeros(ni, dtype=ctype.REAL)
            self._az = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni
        for i in range(ni):
            rx = iobj.rx[i] - jobj.rx
            ry = iobj.ry[i] - jobj.ry
            rz = iobj.rz[i] - jobj.rz
            e2 = iobj.eps2[i] + jobj.eps2
            r2 = rx * rx + ry * ry + rz * rz
            mask = r2 > 0
            inv_r2 = 1 / (r2 + e2)
            inv_r = np.sqrt(inv_r2)
            inv_r3 = inv_r * inv_r2
            inv_r3 *= jobj.mass
            self._ax[i] = -(inv_r3 * rx)[mask].sum()
            self._ay[i] = -(inv_r3 * ry)[mask].sum()
            self._az[i] = -(inv_r3 * rz)[mask].sum()
        return (self._ax[:ni], self._ay[:ni], self._az[:ni])
#    calc = _pycalc


@decallmethods(timings)
class AccJerk(AbstractExtension):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("acc_jerk_kernel", exttype, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes
        lmem = self.kernel.allocate_local_memory(8, np.dtype(ctype.REAL))
        self.kernel.set_args(lmem, start=24)
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self._ax = np.zeros(ni, dtype=ctype.REAL)
            self._ay = np.zeros(ni, dtype=ctype.REAL)
            self._az = np.zeros(ni, dtype=ctype.REAL)
            self._jx = np.zeros(ni, dtype=ctype.REAL)
            self._jy = np.zeros(ni, dtype=ctype.REAL)
            self._jz = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz)
        self._outargs = (self._ax[:ni], self._ay[:ni], self._az[:ni],
                         self._jx[:ni], self._jy[:ni], self._jz[:ni])

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class Tstep(AbstractExtension):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("tstep_kernel", exttype, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real)
        restypes = (cty.c_real_p, cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes
        lmem = self.kernel.allocate_local_memory(8, np.dtype(ctype.REAL))
        self.kernel.set_args(lmem, start=21)
        self.max_output_size = 0

    def set_args(self, iobj, jobj, eta):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self._tstep_a = np.zeros(ni, dtype=ctype.REAL)
            self._tstep_b = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                        eta)
        self._outargs = (self._tstep_a[:ni], self._tstep_b[:ni])

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class PNAcc(AbstractExtension):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("pnacc_kernel", exttype, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint, cty.c_real,
                    cty.c_real, cty.c_real,
                    cty.c_real, cty.c_real,
                    cty.c_real, cty.c_real)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes
        lmem = self.kernel.allocate_local_memory(8, np.dtype(ctype.REAL))
        self.kernel.set_args(lmem, start=29)
        self.max_output_size = 0

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self._pnax = np.zeros(ni, dtype=ctype.REAL)
            self._pnay = np.zeros(ni, dtype=ctype.REAL)
            self._pnaz = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                        clight.pn_order, clight.inv1,
                        clight.inv2, clight.inv3,
                        clight.inv4, clight.inv5,
                        clight.inv6, clight.inv7)
        self._outargs = (self._pnax[:ni], self._pnay[:ni], self._pnaz[:ni])

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class Sakura(AbstractExtension):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("sakura_kernel", exttype, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes
        lmem = self.kernel.allocate_local_memory(8, np.dtype(ctype.REAL))
        self.kernel.set_args(lmem, start=25)
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self._drx = np.zeros(ni, dtype=ctype.REAL)
            self._dry = np.zeros(ni, dtype=ctype.REAL)
            self._drz = np.zeros(ni, dtype=ctype.REAL)
            self._dvx = np.zeros(ni, dtype=ctype.REAL)
            self._dvy = np.zeros(ni, dtype=ctype.REAL)
            self._dvz = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                        dt)
        self._outargs = (self._drx[:ni], self._dry[:ni], self._drz[:ni],
                         self._dvx[:ni], self._dvy[:ni], self._dvz[:ni])

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class NREG_X(AbstractExtension):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("nreg_Xkernel", exttype, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes
        lmem = self.kernel.allocate_local_memory(8, np.dtype(ctype.REAL))
        self.kernel.set_args(lmem, start=26)
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self._rx = np.zeros(ni, dtype=ctype.REAL)
            self._ry = np.zeros(ni, dtype=ctype.REAL)
            self._rz = np.zeros(ni, dtype=ctype.REAL)
            self._ax = np.zeros(ni, dtype=ctype.REAL)
            self._ay = np.zeros(ni, dtype=ctype.REAL)
            self._az = np.zeros(ni, dtype=ctype.REAL)
            self._u = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                        dt)
        self._outargs = (self._rx[:ni], self._ry[:ni], self._rz[:ni],
                         self._ax[:ni], self._ay[:ni], self._az[:ni],
                         self._u[:ni])

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class NREG_V(AbstractExtension):
    """

    """
    def __init__(self, exttype, prec):
        self.kernel = get_kernel("nreg_Vkernel", exttype, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes
        lmem = self.kernel.allocate_local_memory(8, np.dtype(ctype.REAL))
        self.kernel.set_args(lmem[:7], start=21)
        self.max_output_size = 0

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if ni > self.max_output_size:
            self._vx = np.zeros(ni, dtype=ctype.REAL)
            self._vy = np.zeros(ni, dtype=ctype.REAL)
            self._vz = np.zeros(ni, dtype=ctype.REAL)
            self._k = np.zeros(ni, dtype=ctype.REAL)
            self.max_output_size = ni

        self._inargs = (ni,
                        iobj.mass, iobj.vx, iobj.vy, iobj.vz,
                        iobj.ax, iobj.ay, iobj.az,
                        nj,
                        jobj.mass, jobj.vx, jobj.vy, jobj.vz,
                        jobj.ax, jobj.ay, jobj.az,
                        dt)
        self._outargs = (self._vx[:ni], self._vy[:ni], self._vz[:ni],
                         self._k[:ni])

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


exttype = "CL" if "--use_cl" in sys.argv else "C"

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
