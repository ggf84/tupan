# -*- coding: utf-8 -*-
#

"""
This module implements highlevel interfaces for C/CL-extensions.
"""

import logging
from ..config import cli


LOGGER = logging.getLogger(__name__)


class Phi_rectangle(object):
    """

    """
    name = 'phi_kernel_rectangle'

    def set_args(self, ips, jps):
        p = 1
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   jps.n, jps.mass, jps.eps2, jps.rdot[:p])

        outargs = (ips.phi, jps.phi)

        super(Phi_rectangle, self).set_args(inpargs, outargs)


class Phi_triangle(object):
    """

    """
    name = 'phi_kernel_triangle'

    def set_args(self, ips):
        p = 1
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p])

        outargs = (ips.phi,)

        super(Phi_triangle, self).set_args(inpargs, outargs)


class Acc_rectangle(object):
    """

    """
    name = 'acc_kernel_rectangle'

    def set_args(self, ips, jps):
        p = 1
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   jps.n, jps.mass, jps.eps2, jps.rdot[:p])

        outargs = (ips.fdot[:p], jps.fdot[:p])

        super(Acc_rectangle, self).set_args(inpargs, outargs)


class Acc_triangle(object):
    """

    """
    name = 'acc_kernel_triangle'

    def set_args(self, ips):
        p = 1
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p])

        outargs = (ips.fdot[:p],)

        super(Acc_triangle, self).set_args(inpargs, outargs)


class AccJrk_rectangle(object):
    """

    """
    name = 'acc_jrk_kernel_rectangle'

    def set_args(self, ips, jps):
        p = 2
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   jps.n, jps.mass, jps.eps2, jps.rdot[:p])

        outargs = (ips.fdot[:p], jps.fdot[:p])

        super(AccJrk_rectangle, self).set_args(inpargs, outargs)


class AccJrk_triangle(object):
    """

    """
    name = 'acc_jrk_kernel_triangle'

    def set_args(self, ips):
        p = 2
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p])

        outargs = (ips.fdot[:p],)

        super(AccJrk_triangle, self).set_args(inpargs, outargs)


class SnpCrk_rectangle(object):
    """

    """
    name = 'snp_crk_kernel_rectangle'

    def set_args(self, ips, jps):
        p = 4
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   jps.n, jps.mass, jps.eps2, jps.rdot[:p])

        outargs = (ips.fdot[:p], jps.fdot[:p])

        super(SnpCrk_rectangle, self).set_args(inpargs, outargs)


class SnpCrk_triangle(object):
    """

    """
    name = 'snp_crk_kernel_triangle'

    def set_args(self, ips):
        p = 4
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p])

        outargs = (ips.fdot[:p],)

        super(SnpCrk_triangle, self).set_args(inpargs, outargs)


class Tstep_rectangle(object):
    """

    """
    name = 'tstep_kernel_rectangle'

    def set_args(self, ips, jps, eta=None):
        p = 2
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   jps.n, jps.mass, jps.eps2, jps.rdot[:p],
                   eta)

        outargs = (ips.tstep, ips.tstep_sum,
                   jps.tstep, jps.tstep_sum)

        super(Tstep_rectangle, self).set_args(inpargs, outargs)


class Tstep_triangle(object):
    """

    """
    name = 'tstep_kernel_triangle'

    def set_args(self, ips, eta=None):
        p = 2
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   eta)

        outargs = (ips.tstep, ips.tstep_sum)

        super(Tstep_triangle, self).set_args(inpargs, outargs)


class PNAcc_rectangle(object):
    """

    """
    name = 'pnacc_kernel_rectangle'

    def set_args(self, ips, jps, pn=None):
        try:
            clight = self.clight
        except AttributeError:
            pn = pn_struct(pn)
            clight = self.make_struct('CLIGHT', **pn)
            self.clight = clight

        p = 2
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   jps.n, jps.mass, jps.eps2, jps.rdot[:p],
                   clight)

        outargs = (ips.pnacc, jps.pnacc)

        super(PNAcc_rectangle, self).set_args(inpargs, outargs)


class PNAcc_triangle(object):
    """

    """
    name = 'pnacc_kernel_triangle'

    def set_args(self, ips, pn=None):
        try:
            clight = self.clight
        except AttributeError:
            pn = pn_struct(pn)
            clight = self.make_struct('CLIGHT', **pn)
            self.clight = clight

        p = 2
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   clight)

        outargs = (ips.pnacc,)

        super(PNAcc_triangle, self).set_args(inpargs, outargs)


class Sakura_rectangle(object):
    """

    """
    name = 'sakura_kernel_rectangle'

    def set_args(self, ips, jps, dt=None, flag=None):
        p = 2
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   jps.n, jps.mass, jps.eps2, jps.rdot[:p],
                   dt, flag)

        outargs = (ips.drdot, jps.drdot)

        super(Sakura_rectangle, self).set_args(inpargs, outargs)


class Sakura_triangle(object):
    """

    """
    name = 'sakura_kernel_triangle'

    def set_args(self, ips, dt=None, flag=None):
        p = 2
        inpargs = (ips.n, ips.mass, ips.eps2, ips.rdot[:p],
                   dt, flag)

        outargs = (ips.drdot,)

        super(Sakura_triangle, self).set_args(inpargs, outargs)


class Kepler(object):
    """

    """
    name = 'kepler_solver_kernel'

    def set_args(self, ips, jps, dt=None):
        inpargs = (ips.n,
                   ips.mass, ips.rdot[0][0], ips.rdot[0][1], ips.rdot[0][2],
                   ips.eps2, ips.rdot[1][0], ips.rdot[1][1], ips.rdot[1][2],
                   jps.n,
                   jps.mass, jps.rdot[0][0], jps.rdot[0][1], jps.rdot[0][2],
                   jps.eps2, jps.rdot[1][0], jps.rdot[1][1], jps.rdot[1][2],
                   dt)

        outargs = (ips.rdot[0][0], ips.rdot[0][1], ips.rdot[0][2],
                   ips.rdot[1][0], ips.rdot[1][1], ips.rdot[1][2],
                   jps.rdot[0][0], jps.rdot[0][1], jps.rdot[0][2],
                   jps.rdot[1][0], jps.rdot[1][1], jps.rdot[1][2])

        super(Kepler, self).set_args(inpargs, outargs)


def pn_struct(pn):
    order = pn['order']
    clight = pn['clight']
    d = {'order': order}
    for i in range(1, 8):
        d['inv'+str(i)] = clight**(-i)
    return d


def get_backend(backend):
    if backend == 'C':
        from .backend_cffi import CKernel as Kernel
    elif backend == 'CL':
        from .backend_opencl import CLKernel as Kernel
    else:
        msg = "Invalid backend: '{}'. Choices are: ('C', 'CL')"
        raise ValueError(msg.format(backend))
    return Kernel


def make_extension(name, backend):
    cls = globals()[name]
    Kernel = get_backend(backend)

    def __call__(self, *args, **kwargs):
        self.set_args(*args, **kwargs)
        self.run()
        self.map_buffers()
        return args

    Extension = type(name, (cls, Kernel), {'__call__': __call__})
    return Extension(cls.name)


def get_kernel(name, backend=cli.backend):
    if name == 'Kepler':
        kernel = make_extension(name, 'C')

        def func(ips, jps, *args, **kwargs):
            return kernel(ips, jps, *args, **kwargs)

        return func

    kernel_r = make_extension(name+'_rectangle', backend)
    kernel_t = make_extension(name+'_triangle', backend)

    def func(ips, jps, *args, **kwargs):
        if ips != jps:
            return kernel_r(ips, jps, *args, **kwargs)
        return kernel_t(ips, *args, **kwargs)

    return func


# -- End of File --
