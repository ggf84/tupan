# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import abc
import logging
from operator import add, sub
from .base import Base, power_of_two
from ..lib.utils import with_metaclass


LOGGER = logging.getLogger(__name__)
PM = [add, sub]  # plus/minus operator.


class HX(with_metaclass(abc.ABCMeta, object)):
    """

    """
    @property
    @abc.abstractmethod
    def order(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def coefs(self):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def evaluate(ps):
        raise NotImplementedError

    def __init__(self, manager):
        self.initialized = False
        self.cli = manager.cli
        self.update_tstep = manager.update_tstep

    def timestep_criterion(self, ps, eta):
        def A(a2, k):
            return (a2[k-1] * a2[k+1])**0.5 + a2[k]

        order = self.order
        for p in ps.members.values():
            if p.n:
                a2 = (p.rdot[2:2+order]**2).sum(1)
                frac = A(a2, 1) / A(a2, order-2)
                p.tstep[...] = eta * frac**(0.5/(order-3))

    def set_nextstep(self, ps, dt):
        if not self.initialized:
            self.initialized = True
            ps.set_acc_jrk(ps)
            if self.order > 4:
                ps.set_snp_crk(ps)
            if self.update_tstep:
                ps.set_tstep(ps, 0.5 * self.cli.eta)
                return power_of_two(ps, dt)
        else:
            if self.update_tstep:
                self.timestep_criterion(ps, self.cli.eta)
                return power_of_two(ps, dt)
        return dt

    def predict(self, ps, dt):
        """
        Predict the positions and necessary time derivatives of all particles
        at time t to the next time t+dt.
        """
        order = self.order
        nforce = self.nforce
        for p in ps.members.values():
            if p.n:
                for i in range(nforce):
                    drdot = 0
                    for j in reversed(range(1, order-i)):
                        drdot += p.rdot[j+i]
                        drdot *= dt / j
                    p.rdot[i] += drdot
        return ps

    def correct(self, ps1, ps0, dt):
        """
        Correct the positions and velocities of particles based on the new
        values of acceleration and its time derivatives at t+dt and those at
        the previous time t.
        """
        h = dt / 2
        nforce = self.nforce
        coefs = self.coefs[0]
        for p0, p1 in zip(ps0.members.values(), ps1.members.values()):
            if p0.n and p1.n:
                for i in reversed(range(2)):
                    drdot = 0
                    for j in reversed(range(1, nforce)):
                        ff = PM[j % 2](p0.rdot[j+i+1], p1.rdot[j+i+1])
                        drdot += ff * coefs[j]
                        drdot *= h / j
                    ff = PM[0](p0.rdot[i+1], p1.rdot[i+1])
                    drdot += ff * coefs[0]
                    p1.rdot[i] = p0.rdot[i] + h * drdot
        return ps1

    def interpolate(self, ps1, ps0, dt):
        """
        Interpolate polynomial.
        """
        h = dt / 2
        order = self.order
        nforce = self.nforce
        coefs = self.coefs[1]

        hinv = [1.0, 1/h]
        for i in range(2, order):
            hinv.append(i * hinv[1] * hinv[-1])

        for p0, p1 in zip(ps0.members.values(), ps1.members.values()):
            if p0.n and p1.n:
                for i in range(nforce):
                    s = 0
                    c = coefs[i]
                    i += nforce
                    for j in reversed(range(1, nforce)):
                        ff = PM[(i + j) % 2](p0.rdot[j+2], p1.rdot[j+2])
                        s += ff * c[j]
                        s *= h / j
                    ff = PM[i % 2](p0.rdot[2], p1.rdot[2])
                    s += ff * c[0]
                    s *= hinv[i]
                    p1.rdot[i+2] = s

                for i in range(nforce):
                    drdot = 0
                    i += nforce
                    for j in reversed(range(1, order-i)):
                        drdot += p1.rdot[j+i+2]
                        drdot *= h / j
                    p1.rdot[i+2] += drdot

        return ps1

    def pec(self, n, ps0, dt):
        dt = self.set_nextstep(ps0, dt)
        ps1 = self.predict(ps0.copy(), dt)
        while n > 0:
            self.evaluate(ps1)
            ps1 = self.correct(ps1, ps0, dt)
            n -= 1
        ps1 = self.interpolate(ps1, ps0, dt)
        for p1 in ps1.members.values():
            if p1.n:
                p1.time += dt
                p1.nstep += 1
        return ps1


class H2(HX):
    """

    """
    nforce = 1
    order = 2 * nforce
    coefs = [
        [1],
        [
            [-1],
        ],
    ]
    coefs[0] = [1.0/i for i in coefs[0]]
    coefs[1] = [[i/2.0 for i in j] for j in coefs[1]]

    @staticmethod
    def evaluate(ps):
        ps.set_acc(ps)

    def timestep_criterion(self, ps, eta):
        order = self.order
        for p in ps.members.values():
            if p.n:
                a2 = (p.rdot[2:2+order]**2).sum(1)
                p.tstep[...] = eta * (a2[0] / a2[1])**0.5


class H4(HX):
    """

    """
    nforce = 2
    order = 2 * nforce
    coefs = [
        [1, 3],
        [
            [ 0, -1],
            [ 1,  1],
        ],
    ]
    coefs[0] = [1.0/i for i in coefs[0]]
    coefs[1] = [[i/4.0 for i in j] for j in coefs[1]]

    @staticmethod
    def evaluate(ps):
        ps.set_acc_jrk(ps)


class H6(HX):
    """

    """
    nforce = 3
    order = 2 * nforce
    coefs = [
        [1, 5/2., 15/2.],
        [
            [ 10, 10,  4],
            [  0,  1,  2],
            [ -3, -3, -2],
        ],
    ]
    coefs[0] = [1.0/i for i in coefs[0]]
    coefs[1] = [[i/16.0 for i in j] for j in coefs[1]]

    @staticmethod
    def evaluate(ps):
        ps.set_snp_crk(ps)


class H8(HX):
    """

    """
    nforce = 4
    order = 2 * nforce
    coefs = [
        [1, 7/3., 21/4., 35/2.],
        [
            [  0,   5,  10,  6],
            [-21, -21, -16, -6],
            [  0,  -1,  -2, -2],
            [  5,   5,   4,  2],
        ],
    ]
    coefs[0] = [1.0/i for i in coefs[0]]
    coefs[1] = [[i/32.0 for i in j] for j in coefs[1]]

    @staticmethod
    def evaluate(ps):
        ps.set_snp_crk(ps)


class Hermite(Base):
    """

    """
    PROVIDED_METHODS = [
        'c.hermite2', 'a.hermite2',
        'c.hermite4', 'a.hermite4',
        'c.hermite6', 'a.hermite6',
        'c.hermite8', 'a.hermite8',
    ]

    def __init__(self, ps, cli, *args, **kwargs):
        """

        """
        super(Hermite, self).__init__(ps, cli, *args, **kwargs)

        method = cli.method

        if 'c.' in method:
            self.update_tstep = False
            self.shared_tstep = True
        elif 'a.' in method:
            self.update_tstep = True
            self.shared_tstep = True

        if 'hermite2' in method:
            self.hermite = H2(self)
        elif 'hermite4' in method:
            self.hermite = H4(self)
        elif 'hermite6' in method:
            self.hermite = H6(self)
        elif 'hermite8' in method:
            self.hermite = H8(self)

    def do_step(self, ps, dt):
        """

        """
        return self.hermite.pec(1, ps, dt)


# -- End of File --
