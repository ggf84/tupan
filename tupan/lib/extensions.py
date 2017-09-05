# -*- coding: utf-8 -*-
#

"""
This module implements highlevel interfaces for C/CL-extensions.
"""

import logging
from ..config import cli


LOGGER = logging.getLogger(__name__)


class Phi(object):
    """

    """
    name = 'phi_kernel'

    def set_consts(self):
        return []

    def set_bufs(self, ps, nforce=1):
        to_int = self.to_int
        to_buffer = self.to_buffer
        ps.phi[...] = 0
        return [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.phi),
        ]

    def map_bufs(self, bufs, ps, nforce=1):
        self.map_buf(bufs[4], ps.phi)


class Acc(object):
    """

    """
    name = 'acc_kernel'

    def set_consts(self):
        return []

    def set_bufs(self, ps, nforce=1):
        to_int = self.to_int
        to_buffer = self.to_buffer
        ps.fdot[:nforce] = 0
        return [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.fdot[:nforce]),
        ]

    def map_bufs(self, bufs, ps, nforce=1):
        self.map_buf(bufs[4], ps.fdot[:nforce])


class Acc_Jrk(Acc):
    """

    """
    name = 'acc_jrk_kernel'


class Snp_Crk(Acc):
    """

    """
    name = 'snp_crk_kernel'


class Tstep(object):
    """

    """
    name = 'tstep_kernel'

    def set_consts(self, eta=None):
        return [
            self.to_real(eta),
        ]

    def set_bufs(self, ps, nforce=2):
        to_int = self.to_int
        to_buffer = self.to_buffer
        ps.tstep[...] = 0
        ps.tstep_sum[...] = 0
        return [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.tstep),
            to_buffer(ps.tstep_sum),
        ]

    def map_bufs(self, bufs, ps, nforce=2):
        self.map_buf(bufs[4], ps.tstep)
        self.map_buf(bufs[5], ps.tstep_sum)


class PNAcc(object):
    """

    """
    name = 'pnacc_kernel'

    def set_consts(self, order=None, clight=None):
        return [
            self.to_int(order),
            self.to_real(clight),
        ]

    def set_bufs(self, ps, nforce=2):
        to_int = self.to_int
        to_buffer = self.to_buffer
        ps.pnacc[...] = 0
        return [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.pnacc),
        ]

    def map_bufs(self, bufs, ps, nforce=1):
        self.map_buf(bufs[4], ps.pnacc)


class Sakura(object):
    """

    """
    name = 'sakura_kernel'

    def set_consts(self, dt=None, flag=None):
        return [
            self.to_real(dt),
            self.to_int(flag),
        ]

    def set_bufs(self, ps, nforce=2):
        to_int = self.to_int
        to_buffer = self.to_buffer
        ps.drdot[...] = 0
        return [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.drdot),
        ]

    def map_bufs(self, bufs, ps, nforce=1):
        self.map_buf(bufs[4], ps.drdot)


class Kepler(object):
    """

    """
    name = 'kepler_solver_kernel'

    def set_consts(self, dt=None):
        return [
            self.to_real(dt),
        ]

    def set_bufs(self, ps, nforce=2):
        to_int = self.to_int
        to_buffer = self.to_buffer
        return [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[0][0]),
            to_buffer(ps.rdot[0][1]),
            to_buffer(ps.rdot[0][2]),
            to_buffer(ps.rdot[1][0]),
            to_buffer(ps.rdot[1][1]),
            to_buffer(ps.rdot[1][2]),
        ]

    def map_bufs(self, bufs, ps, nforce=2):
        self.map_buf(bufs[3], ps.rdot[0][0])
        self.map_buf(bufs[4], ps.rdot[0][1])
        self.map_buf(bufs[5], ps.rdot[0][2])
        self.map_buf(bufs[6], ps.rdot[1][0])
        self.map_buf(bufs[7], ps.rdot[1][1])
        self.map_buf(bufs[8], ps.rdot[1][2])


def make_extension(name, backend=cli.backend):
    if backend == 'C':
        from .backend_cffi import CKernel as Kernel
    elif backend == 'CL':
        from .backend_opencl import CLKernel as Kernel
    else:
        msg = f"Invalid backend: '{backend}'. Choices are: ('C', 'CL')"
        raise ValueError(msg)

    cls = globals()[name]
    Extension = type(name, (cls, Kernel), {})
    return Extension(cls.name)


# -- End of File --
