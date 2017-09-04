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

    def set_bufs(self, ps, nforce=1):
        to_int = self.to_int
        to_buffer = self.to_buffer
        bufs = [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.phi),
        ]
        return bufs

    def set_args(self, ips):
        ips.phi[...] = 0
        oargs = {
            4: ips.phi,
        }
        bufs = []
        bufs += self.set_bufs(ips)
        return oargs, bufs


class Phi_rectangle(Phi_triangle):
    """

    """
    name = 'phi_kernel_rectangle'

    def set_args(self, ips, jps):
        ips.phi[...] = 0
        jps.phi[...] = 0
        oargs = {
            4: ips.phi,
            9: jps.phi,
        }
        bufs = []
        bufs += self.set_bufs(ips)
        bufs += self.set_bufs(jps)
        return oargs, bufs


class Acc(object):
    """

    """
    name = 'acc_kernel'
    both = True

    def set_bufs(self, ps, nforce=1):
        to_int = self.to_int
        to_buffer = self.to_buffer
        bufs = [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.fdot[:nforce]),
        ]
        return bufs


class Acc_triangle(object):
    """

    """
    name = 'acc_kernel_triangle'

    def set_bufs(self, ps, nforce=1):
        to_int = self.to_int
        to_buffer = self.to_buffer
        bufs = [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.fdot[:nforce]),
        ]
        return bufs

    def set_args(self, ips, nforce=1):
        ips.fdot[:nforce] = 0
        oargs = {
            4: ips.fdot[:nforce],
        }
        bufs = []
        bufs += self.set_bufs(ips, nforce=nforce)
        return oargs, bufs


class Acc_rectangle(Acc_triangle):
    """

    """
    name = 'acc_kernel_rectangle'

    def set_args(self, ips, jps, nforce=1):
        ips.fdot[:nforce] = 0
        jps.fdot[:nforce] = 0
        oargs = {
            4: ips.fdot[:nforce],
            9: jps.fdot[:nforce],
        }
        bufs = []
        bufs += self.set_bufs(ips, nforce=nforce)
        bufs += self.set_bufs(jps, nforce=nforce)
        return oargs, bufs


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

    def set_bufs(self, ps, nforce=2):
        to_int = self.to_int
        to_buffer = self.to_buffer
        bufs = [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.tstep),
            to_buffer(ps.tstep_sum),
        ]
        return bufs

    def set_args(self, ips, eta=None):
        ips.tstep[...] = 0
        ips.tstep_sum[...] = 0
        oargs = {
            5: ips.tstep,
            6: ips.tstep_sum,
        }
        bufs = [
            self.to_real(eta),
        ]
        bufs += self.set_bufs(ips)
        return oargs, bufs


class Tstep_rectangle(Tstep_triangle):
    """

    """
    name = 'tstep_kernel_rectangle'

    def set_args(self, ips, jps, eta=None):
        ips.tstep[...] = 0
        ips.tstep_sum[...] = 0
        ips.tstep[...] = 0
        ips.tstep_sum[...] = 0
        jps.tstep[...] = 0
        jps.tstep_sum[...] = 0
        oargs = {
            5: ips.tstep,
            6: ips.tstep_sum,
            11: jps.tstep,
            12: jps.tstep_sum,
        }
        bufs = [
            self.to_real(eta),
        ]
        bufs += self.set_bufs(ips)
        bufs += self.set_bufs(jps)
        return oargs, bufs


class PNAcc_triangle(object):
    """

    """
    name = 'pnacc_kernel_triangle'

    def set_bufs(self, ps, nforce=2):
        to_int = self.to_int
        to_buffer = self.to_buffer
        bufs = [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.pnacc),
        ]
        return bufs

    def set_args(self, ips, pn=None):
        ips.pnacc[...] = 0
        oargs = {
            6: ips.pnacc,
        }
        bufs = [
            self.to_int(pn['order']),
            self.to_real(pn['clight']),
        ]
        bufs += self.set_bufs(ips)
        return oargs, bufs


class PNAcc_rectangle(PNAcc_triangle):
    """

    """
    name = 'pnacc_kernel_rectangle'

    def set_args(self, ips, jps, pn=None):
        ips.pnacc[...] = 0
        jps.pnacc[...] = 0
        oargs = {
            6: ips.pnacc,
            11: jps.pnacc,
        }
        bufs = [
            self.to_int(pn['order']),
            self.to_real(pn['clight']),
        ]
        bufs += self.set_bufs(ips)
        bufs += self.set_bufs(jps)
        return oargs, bufs


class Sakura_triangle(object):
    """

    """
    name = 'sakura_kernel_triangle'

    def set_bufs(self, ps, nforce=2):
        to_int = self.to_int
        to_buffer = self.to_buffer
        bufs = [
            to_int(ps.n),
            to_buffer(ps.mass),
            to_buffer(ps.eps2),
            to_buffer(ps.rdot[:nforce]),
            to_buffer(ps.drdot),
        ]
        return bufs

    def set_args(self, ips, dt=None, flag=None):
        ips.drdot[...] = 0
        oargs = {
            6: ips.drdot,
        }
        bufs = [
            self.to_real(dt),
            self.to_int(flag),
        ]
        bufs += self.set_bufs(ips)
        return oargs, bufs


class Sakura_rectangle(Sakura_triangle):
    """

    """
    name = 'sakura_kernel_rectangle'

    def set_args(self, ips, jps, dt=None, flag=None):
        ips.drdot[...] = 0
        jps.drdot[...] = 0
        oargs = {
            6: ips.drdot,
            11: jps.drdot,
        }
        bufs = [
            self.to_real(dt),
            self.to_int(flag),
        ]
        bufs += self.set_bufs(ips)
        bufs += self.set_bufs(jps)
        return oargs, bufs


class Kepler(object):
    """

    """
    name = 'kepler_solver_kernel'

    def set_bufs(self, ps, nforce=2):
        to_int = self.to_int
        to_buffer = self.to_buffer
        bufs = [
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
        return bufs

    def set_args(self, ips, jps, dt=None):
        oargs = {
                4: ips.rdot[0][0],
                5: ips.rdot[0][1],
                6: ips.rdot[0][2],
                7: ips.rdot[1][0],
                8: ips.rdot[1][1],
                9: ips.rdot[1][2],
                13: jps.rdot[0][0],
                14: jps.rdot[0][1],
                15: jps.rdot[0][2],
                16: jps.rdot[1][0],
                17: jps.rdot[1][1],
                18: jps.rdot[1][2],
        }
        bufs = [
            self.to_real(dt),
        ]
        bufs += self.set_bufs(ips)
        bufs += self.set_bufs(jps)
        return oargs, bufs


def get_backend(backend):
    if backend == 'C':
        from .backend_cffi import CKernel as Kernel
    elif backend == 'CL':
        from .backend_opencl import CLKernel as Kernel
    else:
        msg = f"Invalid backend: '{backend}'. Choices are: ('C', 'CL')"
        raise ValueError(msg)
    return Kernel


def make_extension(name, backend=cli.backend):
    cls = globals()[name]
    Kernel = get_backend(backend)

    def __call__(self, *args, **kwargs):
        oargs, bufs = self.set_args(*args, **kwargs)
        self.run(*bufs)
        self.map_buffers(oargs, bufs)
        return args

    Extension = type(name, (cls, Kernel), {'__call__': __call__})
    return Extension(cls.name)


# -- End of File --
