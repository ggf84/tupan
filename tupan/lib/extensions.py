# -*- coding: utf-8 -*-
#

"""
This module implements highlevel interfaces for C/CL-extensions.
"""

import logging
from ..config import cli


LOGGER = logging.getLogger(__name__)


class Phi_triangle(object):
    """

    """
    name = 'phi_kernel_triangle'

    def set_args(self, ips):
        p = 1

        to_int = self.to_int
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:p]),
        ]

        ips.phi[...] = 0
        oargs = [
            ips.phi,
        ]

        obufs = [
            to_buffer(ips.phi),
        ]

        return ibufs, (oargs, obufs)


class Phi_rectangle(object):
    """

    """
    name = 'phi_kernel_rectangle'

    def set_args(self, ips, jps):
        p = 1

        to_int = self.to_int
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:p]),
            to_int(jps.n),
            to_buffer(jps.mass),
            to_buffer(jps.eps2),
            to_buffer(jps.rdot[:p]),
        ]

        ips.phi[...] = 0
        jps.phi[...] = 0
        oargs = [
            ips.phi,
            jps.phi,
        ]

        obufs = [
            to_buffer(ips.phi),
            to_buffer(jps.phi),
        ]

        return ibufs, (oargs, obufs)


class Acc_triangle(object):
    """

    """
    name = 'acc_kernel_triangle'

    def set_args(self, ips, nforce=1):
        to_int = self.to_int
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:nforce]),
        ]

        ips.fdot[:nforce] = 0
        oargs = [
            ips.fdot[:nforce],
        ]

        obufs = [
            to_buffer(ips.fdot[:nforce]),
        ]

        return ibufs, (oargs, obufs)


class Acc_rectangle(object):
    """

    """
    name = 'acc_kernel_rectangle'

    def set_args(self, ips, jps, nforce=1):
        to_int = self.to_int
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:nforce]),
            to_int(jps.n),
            to_buffer(jps.mass),
            to_buffer(jps.eps2),
            to_buffer(jps.rdot[:nforce]),
        ]

        ips.fdot[:nforce] = 0
        jps.fdot[:nforce] = 0
        oargs = [
            ips.fdot[:nforce],
            jps.fdot[:nforce],
        ]

        obufs = [
            to_buffer(ips.fdot[:nforce]),
            to_buffer(jps.fdot[:nforce]),
        ]

        return ibufs, (oargs, obufs)


class AccJrk_triangle(Acc_triangle):
    """

    """
    name = 'acc_jrk_kernel_triangle'


class AccJrk_rectangle(Acc_rectangle):
    """

    """
    name = 'acc_jrk_kernel_rectangle'


class SnpCrk_triangle(Acc_triangle):
    """

    """
    name = 'snp_crk_kernel_triangle'


class SnpCrk_rectangle(Acc_rectangle):
    """

    """
    name = 'snp_crk_kernel_rectangle'


class Tstep_triangle(object):
    """

    """
    name = 'tstep_kernel_triangle'

    def set_args(self, ips, eta=None):
        p = 2

        to_int = self.to_int
        to_real = self.to_real
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:p]),
            to_real(eta),
        ]

        ips.tstep[...] = 0
        ips.tstep_sum[...] = 0
        oargs = [
            ips.tstep,
            ips.tstep_sum,
        ]

        obufs = [
            to_buffer(ips.tstep),
            to_buffer(ips.tstep_sum),
        ]

        return ibufs, (oargs, obufs)


class Tstep_rectangle(object):
    """

    """
    name = 'tstep_kernel_rectangle'

    def set_args(self, ips, jps, eta=None):
        p = 2

        to_int = self.to_int
        to_real = self.to_real
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:p]),
            to_int(jps.n),
            to_buffer(jps.mass),
            to_buffer(jps.eps2),
            to_buffer(jps.rdot[:p]),
            to_real(eta),
        ]

        ips.tstep[...] = 0
        ips.tstep_sum[...] = 0
        jps.tstep[...] = 0
        jps.tstep_sum[...] = 0
        oargs = [
            ips.tstep,
            ips.tstep_sum,
            jps.tstep,
            jps.tstep_sum,
        ]

        obufs = [
            to_buffer(ips.tstep),
            to_buffer(ips.tstep_sum),
            to_buffer(jps.tstep),
            to_buffer(jps.tstep_sum),
        ]

        return ibufs, (oargs, obufs)


class PNAcc_triangle(object):
    """

    """
    name = 'pnacc_kernel_triangle'

    def set_args(self, ips, pn=None):
        p = 2

        to_int = self.to_int
        to_real = self.to_real
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:p]),
            to_int(pn['order']),
            to_real(pn['clight']),
        ]

        ips.pnacc[...] = 0
        oargs = [
            ips.pnacc,
        ]

        obufs = [
            to_buffer(ips.pnacc),
        ]

        return ibufs, (oargs, obufs)


class PNAcc_rectangle(object):
    """

    """
    name = 'pnacc_kernel_rectangle'

    def set_args(self, ips, jps, pn=None):
        p = 2

        to_int = self.to_int
        to_real = self.to_real
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:p]),
            to_int(jps.n),
            to_buffer(jps.mass),
            to_buffer(jps.eps2),
            to_buffer(jps.rdot[:p]),
            to_int(pn['order']),
            to_real(pn['clight']),
        ]

        ips.pnacc[...] = 0
        jps.pnacc[...] = 0
        oargs = [
            ips.pnacc,
            jps.pnacc,
        ]

        obufs = [
            to_buffer(ips.pnacc),
            to_buffer(jps.pnacc),
        ]

        return ibufs, (oargs, obufs)


class Sakura_triangle(object):
    """

    """
    name = 'sakura_kernel_triangle'

    def set_args(self, ips, dt=None, flag=None):
        p = 2

        to_int = self.to_int
        to_real = self.to_real
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:p]),
            to_real(dt),
            to_int(flag),
        ]

        ips.drdot[...] = 0
        oargs = [
            ips.drdot,
        ]

        obufs = [
            to_buffer(ips.drdot),
        ]

        return ibufs, (oargs, obufs)


class Sakura_rectangle(object):
    """

    """
    name = 'sakura_kernel_rectangle'

    def set_args(self, ips, jps, dt=None, flag=None):
        p = 2

        to_int = self.to_int
        to_real = self.to_real
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[:p]),
            to_int(jps.n),
            to_buffer(jps.mass),
            to_buffer(jps.eps2),
            to_buffer(jps.rdot[:p]),
            to_real(dt),
            to_int(flag),
        ]

        ips.drdot[...] = 0
        jps.drdot[...] = 0
        oargs = [
            ips.drdot,
            jps.drdot,
        ]

        obufs = [
            to_buffer(ips.drdot),
            to_buffer(jps.drdot),
        ]

        return ibufs, (oargs, obufs)


class Kepler(object):
    """

    """
    name = 'kepler_solver_kernel'

    def set_args(self, ips, jps, dt=None):
        to_int = self.to_int
        to_real = self.to_real
        to_buffer = self.to_buffer

        ibufs = [
            to_int(ips.n),
            to_buffer(ips.mass),
            to_buffer(ips.rdot[0][0]),
            to_buffer(ips.rdot[0][1]),
            to_buffer(ips.rdot[0][2]),
            to_buffer(ips.eps2),
            to_buffer(ips.rdot[1][0]),
            to_buffer(ips.rdot[1][1]),
            to_buffer(ips.rdot[1][2]),
            to_int(jps.n),
            to_buffer(jps.mass),
            to_buffer(jps.rdot[0][0]),
            to_buffer(jps.rdot[0][1]),
            to_buffer(jps.rdot[0][2]),
            to_buffer(jps.eps2),
            to_buffer(jps.rdot[1][0]),
            to_buffer(jps.rdot[1][1]),
            to_buffer(jps.rdot[1][2]),
            to_real(dt),
        ]

        oargs = [
            ips.rdot[0][0],
            ips.rdot[0][1],
            ips.rdot[0][2],
            ips.rdot[1][0],
            ips.rdot[1][1],
            ips.rdot[1][2],
            jps.rdot[0][0],
            jps.rdot[0][1],
            jps.rdot[0][2],
            jps.rdot[1][0],
            jps.rdot[1][1],
            jps.rdot[1][2],
        ]

        obufs = [
            to_buffer(ips.rdot[0][0]),
            to_buffer(ips.rdot[0][1]),
            to_buffer(ips.rdot[0][2]),
            to_buffer(ips.rdot[1][0]),
            to_buffer(ips.rdot[1][1]),
            to_buffer(ips.rdot[1][2]),
            to_buffer(jps.rdot[0][0]),
            to_buffer(jps.rdot[0][1]),
            to_buffer(jps.rdot[0][2]),
            to_buffer(jps.rdot[1][0]),
            to_buffer(jps.rdot[1][1]),
            to_buffer(jps.rdot[1][2]),
        ]

        return ibufs, (oargs, obufs)


def get_backend(backend):
    if backend == 'C':
        from .backend_cffi import CKernel as Kernel
    elif backend == 'CL':
        from .backend_opencl import CLKernel as Kernel
    else:
        msg = "Invalid backend: '{}'. Choices are: ('C', 'CL')"
        raise ValueError(msg.format(backend))
    return Kernel


def make_extension(name, backend=cli.backend):
    cls = globals()[name]
    Kernel = get_backend(backend)

    def __call__(self, *args, **kwargs):
        ibufs, (oargs, obufs) = self.set_args(*args, **kwargs)
        self.run(ibufs+obufs)
        self.map_buffers(oargs, obufs)
        return args

    Extension = type(name, (cls, Kernel), {'__call__': __call__})
    return Extension(cls.name)


# -- End of File --
