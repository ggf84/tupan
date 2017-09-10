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
        to_ibuf = self.to_ibuf
        to_obuf = self.to_obuf
        ps.phi[...] = 0
        return [
            to_int(ps.n),
            to_ibuf(ps.data['mass']),
            to_ibuf(ps.data['eps2']),
            to_ibuf(ps.data['rdot']),
            to_obuf(ps.data['phi']),
        ]

    def map_bufs(self, bufs, ps, nforce=1):
        self.map_buf(ps.data['phi'])


class Acc(object):
    """

    """
    name = 'acc_kernel'

    def set_consts(self):
        return []

    def set_bufs(self, ps, nforce=1):
        to_int = self.to_int
        to_ibuf = self.to_ibuf
        to_obuf = self.to_obuf
        ps.fdot[:nforce] = 0
        return [
            to_int(ps.n),
            to_ibuf(ps.data['mass']),
            to_ibuf(ps.data['eps2']),
            to_ibuf(ps.data['rdot']),
            to_obuf(ps.data['fdot']),
        ]

    def map_bufs(self, bufs, ps, nforce=1):
        self.map_buf(ps.data['fdot'])


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
        to_ibuf = self.to_ibuf
        to_obuf = self.to_obuf
        ps.tstep[...] = 0
        ps.tstep_sum[...] = 0
        return [
            to_int(ps.n),
            to_ibuf(ps.data['mass']),
            to_ibuf(ps.data['eps2']),
            to_ibuf(ps.data['rdot']),
            to_obuf(ps.data['tstep']),
            to_obuf(ps.data['tstep_sum']),
        ]

    def map_bufs(self, bufs, ps, nforce=2):
        self.map_buf(ps.data['tstep'])
        self.map_buf(ps.data['tstep_sum'])


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
        to_ibuf = self.to_ibuf
        to_obuf = self.to_obuf
        ps.pnacc[...] = 0
        return [
            to_int(ps.n),
            to_ibuf(ps.data['mass']),
            to_ibuf(ps.data['eps2']),
            to_ibuf(ps.data['rdot']),
            to_obuf(ps.data['pnacc']),
        ]

    def map_bufs(self, bufs, ps, nforce=1):
        self.map_buf(ps.data['pnacc'])


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
        to_ibuf = self.to_ibuf
        to_obuf = self.to_obuf
        ps.drdot[...] = 0
        return [
            to_int(ps.n),
            to_ibuf(ps.data['mass']),
            to_ibuf(ps.data['eps2']),
            to_ibuf(ps.data['rdot']),
            to_obuf(ps.data['drdot']),
        ]

    def map_bufs(self, bufs, ps, nforce=1):
        self.map_buf(ps.data['drdot'])


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
        to_ibuf = self.to_ibuf
        return [
            to_int(ps.n),
            to_ibuf(ps.mass),
            to_ibuf(ps.eps2),
            to_ibuf(ps.rdot[0][0]),
            to_ibuf(ps.rdot[0][1]),
            to_ibuf(ps.rdot[0][2]),
            to_ibuf(ps.rdot[1][0]),
            to_ibuf(ps.rdot[1][1]),
            to_ibuf(ps.rdot[1][2]),
        ]

    def map_bufs(self, bufs, ps, nforce=2):
        self.map_buf(ps.rdot[0][0])
        self.map_buf(ps.rdot[0][1])
        self.map_buf(ps.rdot[0][2])
        self.map_buf(ps.rdot[1][0])
        self.map_buf(ps.rdot[1][1])
        self.map_buf(ps.rdot[1][2])


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


def make_to_buf(backend=cli.backend):
    if backend == 'C':
        from .backend_cffi import to_cbuf as to_buf
    elif backend == 'CL':
        from .backend_opencl import to_clbuf as to_buf
    else:
        msg = f"Invalid backend: '{backend}'. Choices are: ('C', 'CL')"
        raise ValueError(msg)
    return to_buf


class ArrayWrapper(object):
    """

    """
    def __init__(self, ary):
        self.ptr, self.buf = self.to_buf(ary)
        self.shape = ary.shape
        self.dtype = ary.dtype

    def __getstate__(self):
        return (self.ptr, self.shape, self.dtype)  # buf can not be pickled!

    def __setstate__(self, state):
        ary, self.shape, self.dtype = state
        self.ptr, self.buf = self.to_buf(ary)

    def to_buf(self, ary, to_buf=make_to_buf()):
        return to_buf(ary)


# -- End of File --
