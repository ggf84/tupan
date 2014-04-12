# -*- coding: utf-8 -*-
#

"""This module implements highlevel interfaces for C/CL-extensions.

"""


from __future__ import print_function, division
import sys
import logging
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
           "Kepler", "kepler",
           ]

LOGGER = logging.getLogger(__name__)


@decallmethods(timings)
class PN(object):
    """This class holds the values of the PN parameters.

    """
    def __init__(self):
        self._order = 0
        self._clight = None

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = int(value)

    @property
    def clight(self):
        return self._clight

    @clight.setter
    def clight(self, value):
        self._clight = float(value)


@timings
def get_kernel(name, backend, prec):
    if backend == "C":
        from .cffi_backend import CKernel as Kernel
    elif backend == "CL":
        from .opencl_backend import CLKernel as Kernel
    else:
        msg = "Inappropriate 'backend': {}. Supported values: ['C', 'CL']"
        raise ValueError(msg.format(backend))
    LOGGER.debug(
        "Using '%s' from %s precision %s extension module.",
        name, prec, backend
    )
    return Kernel(prec, name)


class AbstractExtension(object):

    def __init__(self, name, backend, prec):
        self.kernel = get_kernel(name, backend, prec)
        self.inpargs = None
        self.outargs = None

    def set_args(self, ips, jps, **kwargs):
        raise NotImplementedError

    def run(self):
        self.kernel.run()

    def get_result(self):
        return self.kernel.map_buffers(inpargs=self.inpargs,
                                       outargs=self.outargs)

    def calc(self, ips, jps, **kwargs):
        self.set_args(ips, jps, **kwargs)
        self.run()
        return self.get_result()


@decallmethods(timings)
class Phi(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        super(Phi, self).__init__("phi_kernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p)
        outtypes = (cty.c_real_p,)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni, nj)
        if "phi" not in ips.__dict__:
            ips.register_auxiliary_attribute("phi", "real")

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2)
        self.outargs = (ips.phi,)

        self.kernel.args = self.inpargs + self.outargs

    def _pycalc(self, ips, jps):
        # Never use this method for production runs. It is very slow
        # and it's here only for performance comparisons. It is also
        # likely that only the classes Acc and Phi will have an
        # implementation of this method.
        import numpy as np
        ni = ips.n
        if "phi" not in ips.__dict__:
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
        super(Acc, self).__init__("acc_kernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p)
        outtypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni, nj)
        if "ax" not in ips.__dict__:
            ips.register_auxiliary_attribute("ax", "real")
        if "ay" not in ips.__dict__:
            ips.register_auxiliary_attribute("ay", "real")
        if "az" not in ips.__dict__:
            ips.register_auxiliary_attribute("az", "real")

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2)
        self.outargs = (ips.ax, ips.ay, ips.az)

        self.kernel.args = self.inpargs + self.outargs

    def _pycalc(self, ips, jps):
        # Never use this method for production runs. It is very slow
        # and it's here only for performance comparisons. It is also
        # likely that only the classes Acc and Phi will have an
        # implementation of this method.
        import numpy as np
        ni = ips.n
        if "ax" not in ips.__dict__:
            ips.register_auxiliary_attribute("ax", "real")
        if "ay" not in ips.__dict__:
            ips.register_auxiliary_attribute("ay", "real")
        if "az" not in ips.__dict__:
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
        super(AccJerk, self).__init__("acc_jerk_kernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p)
        outtypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni, nj)
        if "ax" not in ips.__dict__:
            ips.register_auxiliary_attribute("ax", "real")
        if "ay" not in ips.__dict__:
            ips.register_auxiliary_attribute("ay", "real")
        if "az" not in ips.__dict__:
            ips.register_auxiliary_attribute("az", "real")
        if "jx" not in ips.__dict__:
            ips.register_auxiliary_attribute("jx", "real")
        if "jy" not in ips.__dict__:
            ips.register_auxiliary_attribute("jy", "real")
        if "jz" not in ips.__dict__:
            ips.register_auxiliary_attribute("jz", "real")

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz)
        self.outargs = (ips.ax, ips.ay, ips.az,
                        ips.jx, ips.jy, ips.jz)

        self.kernel.args = self.inpargs + self.outargs


@decallmethods(timings)
class SnapCrackle(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        super(SnapCrackle, self).__init__("snap_crackle_kernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p)
        outtypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni, nj)
        if "sx" not in ips.__dict__:
            ips.register_auxiliary_attribute("sx", "real")
        if "sy" not in ips.__dict__:
            ips.register_auxiliary_attribute("sy", "real")
        if "sz" not in ips.__dict__:
            ips.register_auxiliary_attribute("sz", "real")
        if "cx" not in ips.__dict__:
            ips.register_auxiliary_attribute("cx", "real")
        if "cy" not in ips.__dict__:
            ips.register_auxiliary_attribute("cy", "real")
        if "cz" not in ips.__dict__:
            ips.register_auxiliary_attribute("cz", "real")

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        ips.ax, ips.ay, ips.az, ips.jx, ips.jy, ips.jz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        jps.ax, jps.ay, jps.az, jps.jx, jps.jy, jps.jz)
        self.outargs = (ips.sx, ips.sy, ips.sz,
                        ips.cx, ips.cy, ips.cz)

        self.kernel.args = self.inpargs + self.outargs


@decallmethods(timings)
class Tstep(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        super(Tstep, self).__init__("tstep_kernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real)
        outtypes = (cty.c_real_p, cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni, nj)
        if "tstep" not in ips.__dict__:
            ips.register_auxiliary_attribute("tstep", "real")
        if "tstepij" not in ips.__dict__:
            ips.register_auxiliary_attribute("tstepij", "real")

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        kwargs['eta'])
        self.outargs = (ips.tstep, ips.tstepij)

        self.kernel.args = self.inpargs + self.outargs


@decallmethods(timings)
class PNAcc(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        super(PNAcc, self).__init__("pnacc_kernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint, cty.c_real,
                    cty.c_real, cty.c_real,
                    cty.c_real, cty.c_real,
                    cty.c_real, cty.c_real)
        outtypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni, nj)
        if "pnax" not in ips.__dict__:
            ips.register_auxiliary_attribute("pnax", "real")
        if "pnay" not in ips.__dict__:
            ips.register_auxiliary_attribute("pnay", "real")
        if "pnaz" not in ips.__dict__:
            ips.register_auxiliary_attribute("pnaz", "real")

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        pn.order, pn.clight**(-1),
                        pn.clight**(-2), pn.clight**(-3),
                        pn.clight**(-4), pn.clight**(-5),
                        pn.clight**(-6), pn.clight**(-7))
        self.outargs = (ips.pnax, ips.pnay, ips.pnaz)

        self.kernel.args = self.inpargs + self.outargs


@decallmethods(timings)
class Sakura(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        super(Sakura, self).__init__("sakura_kernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real, cty.c_int)
        outtypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.vector_width = 1
        self.kernel.set_gsize(ni, nj)
        if "drx" not in ips.__dict__:
            ips.register_auxiliary_attribute("drx", "real")
        if "dry" not in ips.__dict__:
            ips.register_auxiliary_attribute("dry", "real")
        if "drz" not in ips.__dict__:
            ips.register_auxiliary_attribute("drz", "real")
        if "dvx" not in ips.__dict__:
            ips.register_auxiliary_attribute("dvx", "real")
        if "dvy" not in ips.__dict__:
            ips.register_auxiliary_attribute("dvy", "real")
        if "dvz" not in ips.__dict__:
            ips.register_auxiliary_attribute("dvz", "real")

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        kwargs['dt'], kwargs['flag'])
        self.outargs = (ips.drx, ips.dry, ips.drz,
                        ips.dvx, ips.dvy, ips.dvz)

        self.kernel.args = self.inpargs + self.outargs


@decallmethods(timings)
class NREG_X(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        super(NREG_X, self).__init__("nreg_Xkernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real)
        outtypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni, nj)
        if "mrx" not in ips.__dict__:
            ips.register_auxiliary_attribute("mrx", "real")
        if "mry" not in ips.__dict__:
            ips.register_auxiliary_attribute("mry", "real")
        if "mrz" not in ips.__dict__:
            ips.register_auxiliary_attribute("mrz", "real")
        if "ax" not in ips.__dict__:
            ips.register_auxiliary_attribute("ax", "real")
        if "ay" not in ips.__dict__:
            ips.register_auxiliary_attribute("ay", "real")
        if "az" not in ips.__dict__:
            ips.register_auxiliary_attribute("az", "real")
        if "u" not in ips.__dict__:
            ips.register_auxiliary_attribute("u", "real")

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        kwargs['dt'])
        self.outargs = (ips.mrx, ips.mry, ips.mrz,
                        ips.ax, ips.ay, ips.az,
                        ips.u)

        self.kernel.args = self.inpargs + self.outargs


@decallmethods(timings)
class NREG_V(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):
        super(NREG_V, self).__init__("nreg_Vkernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real)
        outtypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.kernel.set_gsize(ni, nj)
        if "mvx" not in ips.__dict__:
            ips.register_auxiliary_attribute("mvx", "real")
        if "mvy" not in ips.__dict__:
            ips.register_auxiliary_attribute("mvy", "real")
        if "mvz" not in ips.__dict__:
            ips.register_auxiliary_attribute("mvz", "real")
        if "mk" not in ips.__dict__:
            ips.register_auxiliary_attribute("mk", "real")

        self.inpargs = (ni,
                        ips.mass, ips.vx, ips.vy, ips.vz,
                        ips.ax, ips.ay, ips.az,
                        nj,
                        jps.mass, jps.vx, jps.vy, jps.vz,
                        jps.ax, jps.ay, jps.az,
                        kwargs['dt'])
        self.outargs = (ips.mvx, ips.mvy, ips.mvz, ips.mk)

        self.kernel.args = self.inpargs + self.outargs


@decallmethods(timings)
class Kepler(AbstractExtension):
    """

    """
    def __init__(self, backend, prec):

        if backend == "CL":    # No need for CL support.
            backend = "C"      # C is fast enough!

        super(Kepler, self).__init__("kepler_solver_kernel", backend, prec)
        cty = self.kernel.cty
        inptypes = (cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_uint,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real)
        outtypes = (cty.c_real_p, cty.c_real_p, cty.c_real_p,
                    cty.c_real_p, cty.c_real_p, cty.c_real_p)
        self.kernel.argtypes = inptypes + outtypes

    def set_args(self, ips, jps, **kwargs):
        ni = ips.n
        nj = jps.n

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        kwargs['dt'])
        self.outargs = (ips.rx, ips.ry, ips.rz,
                        ips.vx, ips.vy, ips.vz)

        self.kernel.args = self.inpargs + self.outargs


BACKEND = "CL" if "--use_cl" in sys.argv else "C"

pn = PN()
phi = Phi(BACKEND, ctype.prec)
acc = Acc(BACKEND, ctype.prec)
acc_jerk = AccJerk(BACKEND, ctype.prec)
snap_crackle = SnapCrackle(BACKEND, ctype.prec)
tstep = Tstep(BACKEND, ctype.prec)
pnacc = PNAcc(BACKEND, ctype.prec)
sakura = Sakura(BACKEND, ctype.prec)
nreg_x = NREG_X(BACKEND, ctype.prec)
nreg_v = NREG_V(BACKEND, ctype.prec)
kepler = Kepler(BACKEND, ctype.prec)


# -- End of File --
