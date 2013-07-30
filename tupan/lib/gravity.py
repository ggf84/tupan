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

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if not "phi" in iobj.__dict__:
            iobj.register_attr("phi", ctype.REAL)

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz)
        self._outargs = (iobj.phi,)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)

    def _pycalc(self, iobj, jobj):
        # Never use this method for production runs. It is very slow
        # and is here only for performance comparisons. It is also
        # likely that only the classes Acc and Phi will have an
        # implementation of this method.
        ni = iobj.n
        if not "phi" in iobj.__dict__:
            iobj.register_attr("phi", ctype.REAL)
        for i in range(ni):
            rx = iobj.rx[i] - jobj.rx
            ry = iobj.ry[i] - jobj.ry
            rz = iobj.rz[i] - jobj.rz
            e2 = iobj.eps2[i] + jobj.eps2
            r2 = rx * rx + ry * ry + rz * rz
            mask = r2 > 0
            inv_r2 = 1 / (r2 + e2)
            inv_r = np.sqrt(inv_r2)
            iobj.phi[i] = -(jobj.mass * inv_r)[mask].sum()
        return (iobj.phi,)
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

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if not "ax" in iobj.__dict__:
            iobj.register_attr("ax", ctype.REAL)
        if not "ay" in iobj.__dict__:
            iobj.register_attr("ay", ctype.REAL)
        if not "az" in iobj.__dict__:
            iobj.register_attr("az", ctype.REAL)

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz)
        self._outargs = (iobj.ax, iobj.ay, iobj.az)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)

    def _pycalc(self, iobj, jobj):
        # Never use this method for production runs. It is very slow
        # and is here only for performance comparisons. It is also
        # likely that only the classes Acc and Phi will have an
        # implementation of this method.
        ni = iobj.n
        if not "ax" in iobj.__dict__:
            iobj.register_attr("ax", ctype.REAL)
        if not "ay" in iobj.__dict__:
            iobj.register_attr("ay", ctype.REAL)
        if not "az" in iobj.__dict__:
            iobj.register_attr("az", ctype.REAL)
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
            iobj.ax[i] = -(inv_r3 * rx)[mask].sum()
            iobj.ay[i] = -(inv_r3 * ry)[mask].sum()
            iobj.az[i] = -(inv_r3 * rz)[mask].sum()
        return (iobj.ax, iobj.ay, iobj.az)
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

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if not "ax" in iobj.__dict__:
            iobj.register_attr("ax", ctype.REAL)
        if not "ay" in iobj.__dict__:
            iobj.register_attr("ay", ctype.REAL)
        if not "az" in iobj.__dict__:
            iobj.register_attr("az", ctype.REAL)
        if not "jx" in iobj.__dict__:
            iobj.register_attr("jx", ctype.REAL)
        if not "jy" in iobj.__dict__:
            iobj.register_attr("jy", ctype.REAL)
        if not "jz" in iobj.__dict__:
            iobj.register_attr("jz", ctype.REAL)

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz)
        self._outargs = (iobj.ax, iobj.ay, iobj.az,
                         iobj.jx, iobj.jy, iobj.jz)

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

    def set_args(self, iobj, jobj, eta):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if not "tstep" in iobj.__dict__:
            iobj.register_attr("tstep", ctype.REAL)
        if not "tstepij" in iobj.__dict__:
            iobj.register_attr("tstepij", ctype.REAL)

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                        eta)
        self._outargs = (iobj.tstep, iobj.tstepij)

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

    def set_args(self, iobj, jobj):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if not "pnax" in iobj.__dict__:
            iobj.register_attr("pnax", ctype.REAL)
        if not "pnay" in iobj.__dict__:
            iobj.register_attr("pnay", ctype.REAL)
        if not "pnaz" in iobj.__dict__:
            iobj.register_attr("pnaz", ctype.REAL)

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
        self._outargs = (iobj.pnax, iobj.pnay, iobj.pnaz)

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

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if not "drx" in iobj.__dict__:
            iobj.register_attr("drx", ctype.REAL)
        if not "dry" in iobj.__dict__:
            iobj.register_attr("dry", ctype.REAL)
        if not "drz" in iobj.__dict__:
            iobj.register_attr("drz", ctype.REAL)
        if not "dvx" in iobj.__dict__:
            iobj.register_attr("dvx", ctype.REAL)
        if not "dvy" in iobj.__dict__:
            iobj.register_attr("dvy", ctype.REAL)
        if not "dvz" in iobj.__dict__:
            iobj.register_attr("dvz", ctype.REAL)

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                        dt)
        self._outargs = (iobj.drx, iobj.dry, iobj.drz,
                         iobj.dvx, iobj.dvy, iobj.dvz)

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

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if not "mrx" in iobj.__dict__:
            iobj.register_attr("mrx", ctype.REAL)
        if not "mry" in iobj.__dict__:
            iobj.register_attr("mry", ctype.REAL)
        if not "mrz" in iobj.__dict__:
            iobj.register_attr("mrz", ctype.REAL)
        if not "ax" in iobj.__dict__:
            iobj.register_attr("ax", ctype.REAL)
        if not "ay" in iobj.__dict__:
            iobj.register_attr("ay", ctype.REAL)
        if not "az" in iobj.__dict__:
            iobj.register_attr("az", ctype.REAL)
        if not "u" in iobj.__dict__:
            iobj.register_attr("u", ctype.REAL)

        self._inargs = (ni,
                        iobj.mass, iobj.rx, iobj.ry, iobj.rz,
                        iobj.eps2, iobj.vx, iobj.vy, iobj.vz,
                        nj,
                        jobj.mass, jobj.rx, jobj.ry, jobj.rz,
                        jobj.eps2, jobj.vx, jobj.vy, jobj.vz,
                        dt)
        self._outargs = (iobj.mrx, iobj.mry, iobj.mrz,
                         iobj.ax, iobj.ay, iobj.az,
                         iobj.u)

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

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.kernel.global_size = ni
        if not "mvx" in iobj.__dict__:
            iobj.register_attr("mvx", ctype.REAL)
        if not "mvy" in iobj.__dict__:
            iobj.register_attr("mvy", ctype.REAL)
        if not "mvz" in iobj.__dict__:
            iobj.register_attr("mvz", ctype.REAL)
        if not "mk" in iobj.__dict__:
            iobj.register_attr("mk", ctype.REAL)

        self._inargs = (ni,
                        iobj.mass, iobj.vx, iobj.vy, iobj.vz,
                        iobj.ax, iobj.ay, iobj.az,
                        nj,
                        jobj.mass, jobj.vx, jobj.vy, jobj.vz,
                        jobj.ax, jobj.ay, jobj.az,
                        dt)
        self._outargs = (iobj.mvx, iobj.mvy, iobj.mvz,
                         iobj.mk)

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
