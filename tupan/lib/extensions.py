# -*- coding: utf-8 -*-
#

"""This module implements highlevel interfaces for C/CL-extensions.

"""


from __future__ import print_function, division
import sys
import logging
from .utils.ctype import Ctype
from .utils.timing import timings, bind_all


__all__ = ['Phi', 'Acc', 'AccJerk', 'SnapCrackle', 'Tstep',
           'PNAcc', 'Sakura', 'NregX', 'NregV', 'Kepler', ]

LOGGER = logging.getLogger(__name__)

BACKEND = 'CL' if '--use_cl' in sys.argv else 'C'
FPWIDTH = Ctype.fpwidth


@bind_all(timings)
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
def get_kernel(name, backend, fpwidth):
    if backend == "C":
        from .cffi_backend import CKernel as Kernel
    elif backend == "CL":
        from .opencl_backend import CLKernel as Kernel
    else:
        msg = "Inappropriate 'backend': {}. Supported values: ['C', 'CL']"
        raise ValueError(msg.format(backend))
    LOGGER.debug(
        "Using '%s' from %s precision %s extension module.",
        name, fpwidth, backend
    )
    return Kernel(fpwidth, name)


class AbstractExtension(object):

    def __init__(self, name, backend, fpwidth):
        self.kernel = get_kernel(name, backend, fpwidth)
        self.inpargs = None
        self.outargs = None

    def set_args(self, ips, jps, **kwargs):
        raise NotImplementedError

    def run(self):
        self.kernel.run()

    def get_result(self):
        return self.kernel.map_buffers(inpargs=self.inpargs,
                                       outargs=self.outargs)

    def __call__(self, ips, jps, **kwargs):
        self.set_args(ips, jps, **kwargs)
        self.run()
        return self.get_result()


@bind_all(timings)
class Phi(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(Phi, self).__init__("phi_kernel", backend, fpwidth)
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
        if not hasattr(ips, 'phi'):
            ips.register_attribute('phi', 'real')

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
        if not hasattr(ips, 'phi'):
            ips.register_attribute('phi', 'real')
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


@bind_all(timings)
class Acc(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(Acc, self).__init__("acc_kernel", backend, fpwidth)
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
        if not hasattr(ips, 'ax'):
            ips.register_attribute('ax', 'real')
        if not hasattr(ips, 'ay'):
            ips.register_attribute('ay', 'real')
        if not hasattr(ips, 'az'):
            ips.register_attribute('az', 'real')

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
        if not hasattr(ips, 'ax'):
            ips.register_attribute('ax', 'real')
        if not hasattr(ips, 'ay'):
            ips.register_attribute('ay', 'real')
        if not hasattr(ips, 'az'):
            ips.register_attribute('az', 'real')
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


@bind_all(timings)
class AccJerk(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(AccJerk, self).__init__("acc_jerk_kernel", backend, fpwidth)
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
        if not hasattr(ips, 'ax'):
            ips.register_attribute('ax', 'real')
        if not hasattr(ips, 'ay'):
            ips.register_attribute('ay', 'real')
        if not hasattr(ips, 'az'):
            ips.register_attribute('az', 'real')
        if not hasattr(ips, 'jx'):
            ips.register_attribute('jx', 'real')
        if not hasattr(ips, 'jy'):
            ips.register_attribute('jy', 'real')
        if not hasattr(ips, 'jz'):
            ips.register_attribute('jz', 'real')

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz)
        self.outargs = (ips.ax, ips.ay, ips.az,
                        ips.jx, ips.jy, ips.jz)

        self.kernel.args = self.inpargs + self.outargs


@bind_all(timings)
class SnapCrackle(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(SnapCrackle, self).__init__("snap_crackle_kernel",
                                          backend, fpwidth)
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
        if not hasattr(ips, 'sx'):
            ips.register_attribute('sx', 'real')
        if not hasattr(ips, 'sy'):
            ips.register_attribute('sy', 'real')
        if not hasattr(ips, 'sz'):
            ips.register_attribute('sz', 'real')
        if not hasattr(ips, 'cx'):
            ips.register_attribute('cx', 'real')
        if not hasattr(ips, 'cy'):
            ips.register_attribute('cy', 'real')
        if not hasattr(ips, 'cz'):
            ips.register_attribute('cz', 'real')

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


@bind_all(timings)
class Tstep(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(Tstep, self).__init__("tstep_kernel", backend, fpwidth)
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
        if not hasattr(ips, 'tstep'):
            ips.register_attribute('tstep', 'real')
        if not hasattr(ips, 'tstepij'):
            ips.register_attribute('tstepij', 'real')

        self.inpargs = (ni,
                        ips.mass, ips.rx, ips.ry, ips.rz,
                        ips.eps2, ips.vx, ips.vy, ips.vz,
                        nj,
                        jps.mass, jps.rx, jps.ry, jps.rz,
                        jps.eps2, jps.vx, jps.vy, jps.vz,
                        kwargs['eta'])
        self.outargs = (ips.tstep, ips.tstepij)

        self.kernel.args = self.inpargs + self.outargs


@bind_all(timings)
class PNAcc(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(PNAcc, self).__init__("pnacc_kernel", backend, fpwidth)
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
        if not hasattr(ips, 'pnax'):
            ips.register_attribute('pnax', 'real')
        if not hasattr(ips, 'pnay'):
            ips.register_attribute('pnay', 'real')
        if not hasattr(ips, 'pnaz'):
            ips.register_attribute('pnaz', 'real')

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


@bind_all(timings)
class Sakura(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(Sakura, self).__init__("sakura_kernel", backend, fpwidth)
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
        if not hasattr(ips, 'drx'):
            ips.register_attribute('drx', 'real')
        if not hasattr(ips, 'dry'):
            ips.register_attribute('dry', 'real')
        if not hasattr(ips, 'drz'):
            ips.register_attribute('drz', 'real')
        if not hasattr(ips, 'dvx'):
            ips.register_attribute('dvx', 'real')
        if not hasattr(ips, 'dvy'):
            ips.register_attribute('dvy', 'real')
        if not hasattr(ips, 'dvz'):
            ips.register_attribute('dvz', 'real')

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


@bind_all(timings)
class NregX(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(NregX, self).__init__("nreg_Xkernel", backend, fpwidth)
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
        if not hasattr(ips, 'mrx'):
            ips.register_attribute('mrx', 'real')
        if not hasattr(ips, 'mry'):
            ips.register_attribute('mry', 'real')
        if not hasattr(ips, 'mrz'):
            ips.register_attribute('mrz', 'real')
        if not hasattr(ips, 'ax'):
            ips.register_attribute('ax', 'real')
        if not hasattr(ips, 'ay'):
            ips.register_attribute('ay', 'real')
        if not hasattr(ips, 'az'):
            ips.register_attribute('az', 'real')
        if not hasattr(ips, 'u'):
            ips.register_attribute('u', 'real')

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


@bind_all(timings)
class NregV(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):
        super(NregV, self).__init__("nreg_Vkernel", backend, fpwidth)
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
        if not hasattr(ips, 'mvx'):
            ips.register_attribute('mvx', 'real')
        if not hasattr(ips, 'mvy'):
            ips.register_attribute('mvy', 'real')
        if not hasattr(ips, 'mvz'):
            ips.register_attribute('mvz', 'real')
        if not hasattr(ips, 'mk'):
            ips.register_attribute('mk', 'real')

        self.inpargs = (ni,
                        ips.mass, ips.vx, ips.vy, ips.vz,
                        ips.ax, ips.ay, ips.az,
                        nj,
                        jps.mass, jps.vx, jps.vy, jps.vz,
                        jps.ax, jps.ay, jps.az,
                        kwargs['dt'])
        self.outargs = (ips.mvx, ips.mvy, ips.mvz, ips.mk)

        self.kernel.args = self.inpargs + self.outargs


@bind_all(timings)
class Kepler(AbstractExtension):
    """

    """
    def __init__(self, backend=BACKEND, fpwidth=FPWIDTH):

        if backend == "CL":    # No need for CL support.
            backend = "C"      # C is fast enough!

        super(Kepler, self).__init__("kepler_solver_kernel", backend, fpwidth)
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


pn = PN()


# -- End of File --
