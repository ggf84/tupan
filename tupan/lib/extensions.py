# -*- coding: utf-8 -*-
#

"""This module implements highlevel interfaces for C/CL-extensions.

"""


from __future__ import print_function, division
import sys
import logging
import numpy as np
from .utils import ctype
from .utils.timing import decallmethods, timings


__all__ = ["Phi", "phi",
           "Acc", "acc",
           "AccJerk", "acc_jerk",
           "SnapCrackle", "snap_crackle",
           "Tstep", "tstep",
           "PNAcc", "pnacc",
           "Sakura", "sakura",
           "NREG_X", "nreg_x",
           "NREG_V", "nreg_v",
           ]

logger = logging.getLogger(__name__)


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


@timings
def get_kernel(name, backend, prec):
    if backend == "C":
        from .cffi_backend import CKernel as Kernel
    elif backend == "CL":
        from .opencl_backend import CLKernel as Kernel
    else:
        msg = "Inappropriate 'backend': {}. Supported values: ['C', 'CL']"
        raise ValueError(msg.format(backend))
    logger.debug(
        "Using '%s' from %s precision %s extension module.",
        name, prec, backend
    )
    return Kernel(prec, name)


@timings
def prepare_args(args, argtypes):
    return [argtype(arg) for (arg, argtype) in zip(args, argtypes)]


class AbstractExtension(object):
    def set_args(self, ips, jps, *args):
        raise NotImplemented

    def run(self):
        self.kernel.run()

    def get_result(self):
        return self.kernel.map_buffers(self._outargs, self.outargs)

    def calc(self, ips, jps, *args):
        self.set_args(ips, jps, *args)
        self.run()
        return self.get_result()


@decallmethods(timings)
class Phi(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("phi_kernel", backend, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p)
        restypes = (cty.c_real_p,)
        self.argtypes = argtypes
        self.restypes = restypes

    def set_args(self, ips, jps):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni)
        if not "phi" in ips.__dict__:
            ips.register_auxiliary_attribute("phi", "real")

        self._inargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2)
        self._outargs = (ips.phi,)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)

    def _pycalc(self, ips, jps):
        # Never use this method for production runs. It is very slow
        # and is here only for performance comparisons. It is also
        # likely that only the classes Acc and Phi will have an
        # implementation of this method.
        ni = ips.n
        if not "phi" in ips.__dict__:
            ips.register_auxiliary_attribute("phi", "real")
        for i in range(ni):
            rx = ips.rx[i] - jps.rx
            ry = ips.ry[i] - jps.ry
            rz = ips.rz[i] - jps.rz
            e2 = ips.eps2[i] + jps.eps2
            r2 = rx * rx + ry * ry + rz * rz
            mask = r2 > 0
            inv_r2 = 1 / (r2 + e2)
            inv_r = np.sqrt(inv_r2)
            ips.phi[i] = -(jps.mass * inv_r)[mask].sum()
        return (ips.phi,)
#    calc = _pycalc


@decallmethods(timings)
class Acc(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("acc_kernel", backend, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes

    def set_args(self, ips, jps):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni)
        if not "ax" in ips.__dict__:
            ips.register_auxiliary_attribute("ax", "real")
        if not "ay" in ips.__dict__:
            ips.register_auxiliary_attribute("ay", "real")
        if not "az" in ips.__dict__:
            ips.register_auxiliary_attribute("az", "real")

        self._inargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2)
        self._outargs = (ips.ax, ips.ay, ips.az)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)

    def _pycalc(self, ips, jps):
        # Never use this method for production runs. It is very slow
        # and is here only for performance comparisons. It is also
        # likely that only the classes Acc and Phi will have an
        # implementation of this method.
        ni = ips.n
        if not "ax" in ips.__dict__:
            ips.register_auxiliary_attribute("ax", "real")
        if not "ay" in ips.__dict__:
            ips.register_auxiliary_attribute("ay", "real")
        if not "az" in ips.__dict__:
            ips.register_auxiliary_attribute("az", "real")
        for i in range(ni):
            rx = ips.rx[i] - jps.rx
            ry = ips.ry[i] - jps.ry
            rz = ips.rz[i] - jps.rz
            e2 = ips.eps2[i] + jps.eps2
            r2 = rx * rx + ry * ry + rz * rz
            mask = r2 > 0
            inv_r2 = 1 / (r2 + e2)
            inv_r = np.sqrt(inv_r2)
            inv_r3 = inv_r * inv_r2
            inv_r3 *= jps.mass
            ips.ax[i] = -(inv_r3 * rx)[mask].sum()
            ips.ay[i] = -(inv_r3 * ry)[mask].sum()
            ips.az[i] = -(inv_r3 * rz)[mask].sum()
        return (ips.ax, ips.ay, ips.az)
#    calc = _pycalc


@decallmethods(timings)
class AccJerk(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("acc_jerk_kernel", backend, prec)
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

    def set_args(self, ips, jps):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni)
        if not "ax" in ips.__dict__:
            ips.register_auxiliary_attribute("ax", "real")
        if not "ay" in ips.__dict__:
            ips.register_auxiliary_attribute("ay", "real")
        if not "az" in ips.__dict__:
            ips.register_auxiliary_attribute("az", "real")
        if not "jx" in ips.__dict__:
            ips.register_auxiliary_attribute("jx", "real")
        if not "jy" in ips.__dict__:
            ips.register_auxiliary_attribute("jy", "real")
        if not "jz" in ips.__dict__:
            ips.register_auxiliary_attribute("jz", "real")

        self._inargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz)
        self._outargs = (ips.ax, ips.ay, ips.az,
                         ips.jx, ips.jy, ips.jz)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class SnapCrackle(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("snap_crackle_kernel", backend, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes

    def set_args(self, ips, jps):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni)
        if not "sx" in ips.__dict__:
            ips.register_auxiliary_attribute("sx", "real")
        if not "sy" in ips.__dict__:
            ips.register_auxiliary_attribute("sy", "real")
        if not "sz" in ips.__dict__:
            ips.register_auxiliary_attribute("sz", "real")
        if not "cx" in ips.__dict__:
            ips.register_auxiliary_attribute("cx", "real")
        if not "cy" in ips.__dict__:
            ips.register_auxiliary_attribute("cy", "real")
        if not "cz" in ips.__dict__:
            ips.register_auxiliary_attribute("cz", "real")

        self._inargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        ips.ax, ips.ay, ips.az, ips.jx, ips.jy, ips.jz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        jps.ax, jps.ay, jps.az, jps.jx, jps.jy, jps.jz)
        self._outargs = (ips.sx, ips.sy, ips.sz,
                         ips.cx, ips.cy, ips.cz)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class Tstep(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("tstep_kernel", backend, prec)
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

    def set_args(self, ips, jps, eta):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni)
        if not "tstep" in ips.__dict__:
            ips.register_auxiliary_attribute("tstep", "real")
        if not "tstepij" in ips.__dict__:
            ips.register_auxiliary_attribute("tstepij", "real")

        self._inargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        eta)
        self._outargs = (ips.tstep, ips.tstepij)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class PNAcc(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("pnacc_kernel", backend, prec)
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

    def set_args(self, ips, jps):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni)
        if not "pnax" in ips.__dict__:
            ips.register_auxiliary_attribute("pnax", "real")
        if not "pnay" in ips.__dict__:
            ips.register_auxiliary_attribute("pnay", "real")
        if not "pnaz" in ips.__dict__:
            ips.register_auxiliary_attribute("pnaz", "real")

        self._inargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        clight.pn_order, clight.inv1,
                        clight.inv2, clight.inv3,
                        clight.inv4, clight.inv5,
                        clight.inv6, clight.inv7)
        self._outargs = (ips.pnax, ips.pnay, ips.pnaz)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class Sakura(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("sakura_kernel", backend, prec)
        cty = self.kernel.cty
        argtypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real, cty.c_int)
        restypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.argtypes = argtypes
        self.restypes = restypes

    def set_args(self, ips, jps, dt, flag):
        ni = ips.n
        nj = jps.n

#        self.kernel.set_gsize(ni)
        vw = 1
        gsize = ((ni + 2 * vw - 1) // (2 * vw)) * 2
        self.kernel.global_size = (gsize, 1, 1)
        if not "drx" in ips.__dict__:
            ips.register_auxiliary_attribute("drx", "real")
        if not "dry" in ips.__dict__:
            ips.register_auxiliary_attribute("dry", "real")
        if not "drz" in ips.__dict__:
            ips.register_auxiliary_attribute("drz", "real")
        if not "dvx" in ips.__dict__:
            ips.register_auxiliary_attribute("dvx", "real")
        if not "dvy" in ips.__dict__:
            ips.register_auxiliary_attribute("dvy", "real")
        if not "dvz" in ips.__dict__:
            ips.register_auxiliary_attribute("dvz", "real")

        self._inargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        dt, flag)
        self._outargs = (ips.drx, ips.dry, ips.drz,
                         ips.dvx, ips.dvy, ips.dvz)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class NREG_X(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("nreg_Xkernel", backend, prec)
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

    def set_args(self, ips, jps, dt):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni)
        if not "mrx" in ips.__dict__:
            ips.register_auxiliary_attribute("mrx", "real")
        if not "mry" in ips.__dict__:
            ips.register_auxiliary_attribute("mry", "real")
        if not "mrz" in ips.__dict__:
            ips.register_auxiliary_attribute("mrz", "real")
        if not "ax" in ips.__dict__:
            ips.register_auxiliary_attribute("ax", "real")
        if not "ay" in ips.__dict__:
            ips.register_auxiliary_attribute("ay", "real")
        if not "az" in ips.__dict__:
            ips.register_auxiliary_attribute("az", "real")
        if not "u" in ips.__dict__:
            ips.register_auxiliary_attribute("u", "real")

        self._inargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        dt)
        self._outargs = (ips.mrx, ips.mry, ips.mrz,
                         ips.ax, ips.ay, ips.az,
                         ips.u)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


@decallmethods(timings)
class NREG_V(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        self.kernel = get_kernel("nreg_Vkernel", backend, prec)
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

    def set_args(self, ips, jps, dt):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni)
        if not "mvx" in ips.__dict__:
            ips.register_auxiliary_attribute("mvx", "real")
        if not "mvy" in ips.__dict__:
            ips.register_auxiliary_attribute("mvy", "real")
        if not "mvz" in ips.__dict__:
            ips.register_auxiliary_attribute("mvz", "real")
        if not "mk" in ips.__dict__:
            ips.register_auxiliary_attribute("mk", "real")

        self._inargs = (ni,
                        ips.mass, ips.vx, ips.vy, ips.vz,
                        ips.ax, ips.ay, ips.az,
                        nj,
                        jps.mass, jps.vx, jps.vy, jps.vz,
                        jps.ax, jps.ay, jps.az,
                        dt)
        self._outargs = (ips.mvx, ips.mvy, ips.mvz,
                         ips.mk)

        self.inargs = prepare_args(self._inargs, self.argtypes)
        self.outargs = prepare_args(self._outargs, self.restypes)

        self.kernel.set_args(self.inargs + self.outargs)


backend = "CL" if "--use_cl" in sys.argv else "C"

clight = Clight()
phi = Phi(backend, ctype.prec)
acc = Acc(backend, ctype.prec)
acc_jerk = AccJerk(backend, ctype.prec)
snap_crackle = SnapCrackle(backend, ctype.prec)
tstep = Tstep(backend, ctype.prec)
pnacc = PNAcc(backend, ctype.prec)
sakura = Sakura(backend, ctype.prec)
nreg_x = NREG_X(backend, ctype.prec)
nreg_v = NREG_V(backend, ctype.prec)


########## end of file ##########
