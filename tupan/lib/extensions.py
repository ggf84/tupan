# -*- coding: utf-8 -*-
#

"""This module implements highlevel interfaces for C/CL-extensions.

"""


from __future__ import print_function, division
import logging
from ..config import options
from .utils.timing import timings, bind_all


__all__ = ['Phi', 'Acc', 'AccJerk', 'SnapCrackle', 'Tstep',
           'PNAcc', 'Sakura', 'NregX', 'NregV', 'Kepler', ]

LOGGER = logging.getLogger(__name__)


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


class AbstractExtension(object):
    """

    """
    def __init__(self, name, backend):
        if backend == 'C':
            from .backend_cffi import drv
        elif backend == 'CL':
            from .backend_opencl import drv
        else:
            msg = "backend: invalid choice: '{}' (choose from 'C', 'CL')"
            raise ValueError(msg.format(backend))
        self.kernel = drv.get_kernel(name)

    def set_args(self, ips, jps, **kwargs):
        raise NotImplementedError

    def run(self):
        self.kernel.run()

    def get_result(self):
        return self.kernel.map_buffers()

    def __call__(self, ips, jps, **kwargs):
        self.set_args(ips, jps, **kwargs)
        self.run()
        return self.get_result()


@bind_all(timings)
class Phi(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(Phi, self).__init__('phi_kernel', backend)

    def set_args(self, ips, jps, **kwargs):
        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2,
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2)

        outargs = (ips.phi,)

        self.kernel.set_args(inpargs, outargs)

    def _pycalc(self, ips, jps):
        # Never use this method for production runs. It is very
        # slow and it's here only for performance comparisons.
        import numpy as np
        for i in range(ips.n):
            r = (ips.pos[..., i] - jps.pos.T).T
            e2 = ips.eps2[i] + jps.eps2
            r2 = (r**2).sum(0)
            mask = r2 > 0
            inv_r2 = 1 / (r2 + e2)
            inv_r = np.sqrt(inv_r2)
            ips.phi[i] = -(jps.mass * inv_r)[mask].sum(0)
        return (ips.phi,)
#    __call__ = _pycalc


@bind_all(timings)
class Acc(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(Acc, self).__init__('acc_kernel', backend)

    def set_args(self, ips, jps, **kwargs):
        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2,
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2)

        outargs = (ips.acc[0], ips.acc[1], ips.acc[2])

        self.kernel.set_args(inpargs, outargs)

    def _pycalc(self, ips, jps):
        # Never use this method for production runs. It is very
        # slow and it's here only for performance comparisons.
        import numpy as np
        for i in range(ips.n):
            r = (ips.pos[..., i] - jps.pos.T).T
            e2 = ips.eps2[i] + jps.eps2
            r2 = (r**2).sum(0)
            mask = r2 > 0
            inv_r2 = 1 / (r2 + e2)
            inv_r = np.sqrt(inv_r2)
            inv_r3 = jps.mass * inv_r * inv_r2
            ips.acc[..., i] = -(inv_r3 * r).T[mask].sum(0)
        return (ips.acc[0], ips.acc[1], ips.acc[2])
#    __call__ = _pycalc


@bind_all(timings)
class AccJerk(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(AccJerk, self).__init__('acc_jerk_kernel', backend)

    def set_args(self, ips, jps, **kwargs):
        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2, ips.vel[0], ips.vel[1], ips.vel[2],
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2, jps.vel[0], jps.vel[1], jps.vel[2])

        outargs = (ips.acc[0], ips.acc[1], ips.acc[2],
                   ips.jrk[0], ips.jrk[1], ips.jrk[2])

        self.kernel.set_args(inpargs, outargs)


@bind_all(timings)
class SnapCrackle(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(SnapCrackle, self).__init__('snap_crackle_kernel', backend)

    def set_args(self, ips, jps, **kwargs):
        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2, ips.vel[0], ips.vel[1], ips.vel[2],
                   ips.acc[0], ips.acc[1], ips.acc[2],
                   ips.jrk[0], ips.jrk[1], ips.jrk[2],
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2, jps.vel[0], jps.vel[1], jps.vel[2],
                   jps.acc[0], jps.acc[1], jps.acc[2],
                   jps.jrk[0], jps.jrk[1], jps.jrk[2])

        outargs = (ips.snp[0], ips.snp[1], ips.snp[2],
                   ips.crk[0], ips.crk[1], ips.crk[2])

        self.kernel.set_args(inpargs, outargs)


@bind_all(timings)
class Tstep(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(Tstep, self).__init__('tstep_kernel', backend)

    def set_args(self, ips, jps, **kwargs):
        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2, ips.vel[0], ips.vel[1], ips.vel[2],
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2, jps.vel[0], jps.vel[1], jps.vel[2],
                   kwargs['eta'])

        outargs = (ips.tstep, ips.tstepij)

        self.kernel.set_args(inpargs, outargs)


@bind_all(timings)
class PNAcc(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(PNAcc, self).__init__('pnacc_kernel', backend)

    def set_args(self, ips, jps, **kwargs):
        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2, ips.vel[0], ips.vel[1], ips.vel[2],
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2, jps.vel[0], jps.vel[1], jps.vel[2],
                   pn.order, pn.clight**(-1),
                   pn.clight**(-2), pn.clight**(-3),
                   pn.clight**(-4), pn.clight**(-5),
                   pn.clight**(-6), pn.clight**(-7))

        outargs = (ips.pnacc[0], ips.pnacc[1], ips.pnacc[2])

        self.kernel.set_args(inpargs, outargs)


@bind_all(timings)
class Sakura(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(Sakura, self).__init__('sakura_kernel', backend)

    def set_args(self, ips, jps, **kwargs):
        if not hasattr(ips, 'dpos'):
            ips.register_attribute('dpos', (3,), 'real')
        if not hasattr(ips, 'dvel'):
            ips.register_attribute('dvel', (3,), 'real')

        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2, ips.vel[0], ips.vel[1], ips.vel[2],
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2, jps.vel[0], jps.vel[1], jps.vel[2],
                   kwargs['dt'], kwargs['flag'])

        outargs = (ips.dpos[0], ips.dpos[1], ips.dpos[2],
                   ips.dvel[0], ips.dvel[1], ips.dvel[2])

        self.kernel.set_args(inpargs, outargs)


@bind_all(timings)
class NregX(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(NregX, self).__init__('nreg_Xkernel', backend)

    def set_args(self, ips, jps, **kwargs):
        if not hasattr(ips, 'mr'):
            ips.register_attribute('mr', (3,), 'real')
        if not hasattr(ips, 'u'):
            ips.register_attribute('u', (), 'real')

        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2, ips.vel[0], ips.vel[1], ips.vel[2],
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2, jps.vel[0], jps.vel[1], jps.vel[2],
                   kwargs['dt'])

        outargs = (ips.mr[0], ips.mr[1], ips.mr[2],
                   ips.acc[0], ips.acc[1], ips.acc[2],
                   ips.u)

        self.kernel.set_args(inpargs, outargs)


@bind_all(timings)
class NregV(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):
        super(NregV, self).__init__('nreg_Vkernel', backend)

    def set_args(self, ips, jps, **kwargs):
        if not hasattr(ips, 'mv'):
            ips.register_attribute('mv', (3,), 'real')
        if not hasattr(ips, 'mk'):
            ips.register_attribute('mk', (), 'real')

        inpargs = (ips.n,
                   ips.mass, ips.vel[0], ips.vel[1], ips.vel[2],
                   ips.acc[0], ips.acc[1], ips.acc[2],
                   jps.n,
                   jps.mass, jps.vel[0], jps.vel[1], jps.vel[2],
                   jps.acc[0], jps.acc[1], jps.acc[2],
                   kwargs['dt'])

        outargs = (ips.mv[0], ips.mv[1], ips.mv[2], ips.mk)

        self.kernel.set_args(inpargs, outargs)


@bind_all(timings)
class Kepler(AbstractExtension):
    """

    """
    def __init__(self, backend=options.backend):

        if backend == 'CL':    # No need for CL support.
            backend = 'C'      # C is fast enough!

        super(Kepler, self).__init__('kepler_solver_kernel', backend)

    def set_args(self, ips, jps, **kwargs):
        inpargs = (ips.n,
                   ips.mass, ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.eps2, ips.vel[0], ips.vel[1], ips.vel[2],
                   jps.n,
                   jps.mass, jps.pos[0], jps.pos[1], jps.pos[2],
                   jps.eps2, jps.vel[0], jps.vel[1], jps.vel[2],
                   kwargs['dt'])

        outargs = (ips.pos[0], ips.pos[1], ips.pos[2],
                   ips.vel[0], ips.vel[1], ips.vel[2])

        self.kernel.set_args(inpargs, outargs)


pn = PN()


# -- End of File --
