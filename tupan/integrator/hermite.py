# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
from operator import add, sub
from abc import ABCMeta, abstractmethod
from .base import Base, power_of_two
from ..lib.utils import with_metaclass
from ..lib.utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)
PM = [add, sub]  # plus/minus operator.


class HX(with_metaclass(ABCMeta, object)):
    """

    """
    @property
    @abstractmethod
    def order(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def coefs(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def evaluate(ps):
        raise NotImplementedError

    def __init__(self, manager):
        self.initialized = False
        self.eta = manager.eta
        self.update_tstep = manager.update_tstep

    def timestep_criterion(self, ps, eta):
        def A(a2, k):
            return (a2[k-1] * a2[k+1])**0.5 + a2[k]

        p = self.order
        a2 = (ps.rdot[2:p+2]**2).sum(1)
        ps.tstep[...] = eta * (A(a2, 1) / A(a2, p-2))**(0.5/(p-3))

    def set_nextstep(self, ps, dt):
        if not self.initialized:
            self.initialized = True
            ps.set_acc_jrk(ps)
            if self.order > 4:
                ps.set_snp_crk(ps)
            if self.update_tstep:
                ps.set_tstep(ps, self.eta)
                return power_of_two(ps, dt)
        else:
            if self.update_tstep:
                self.timestep_criterion(ps, self.eta)
                return power_of_two(ps, dt)
        return dt

    def predict(self, ps, dt):
        """
        Predict the positions and necessary time derivatives of all particles
        at time t to the next time t+dt.
        """
        order = self.order
        for i in range(order//2):
            drdot = 0
            for j in reversed(range(i+1, order)):
                drdot += ps.rdot[j]
                drdot *= dt / (j-i)
            ps.rdot[i] += drdot
        return ps

    def correct(self, ps1, ps0, dt):
        """
        Correct the positions and velocities of particles based on the new
        values of acceleration and its time derivatives at t+dt and those at
        the previous time t.
        """
        order = self.order
        coefs = self.coefs[0]

        h = dt / 2
        for i in reversed(range(2)):
            drdot = 0
            for j in reversed(range(order//2)):
                k = j % 2
                c = coefs[j]
                ff = PM[k](ps0.rdot[j+i+1], ps1.rdot[j+i+1])
                j += 1 if j == 0 else 0
                drdot += c * ff
                drdot *= h / j
            ps1.rdot[i, ...] = ps0.rdot[i] + drdot
        return ps1

    def interpolate(self, ps1, ps0, dt):
        """
        Interpolate polynomial.
        """
        order = self.order
        coefs = self.coefs[1]

        h = dt / 2
        hinv = [1.0, 1/h]
        for i in range(2, order):
            hinv.append(i * hinv[1] * hinv[-1])

        p = order//2
        A = [
            ps0.rdot[2:2+p] + ps1.rdot[2:2+p],
            ps0.rdot[2:2+p] - ps1.rdot[2:2+p],
        ]

        for i in reversed(range(order//2)):
            s = 0
            for j in reversed(range(order//2)):
                c = coefs[i][j]
                if c != 0:
                    k = (i+j+order//2) % 2
                    s += c * A[k][j]
                if j != 0:
                    s *= h / j
            i += order//2
            ps1.rdot[i+2, ...] = s * hinv[i]

        for i in range(order//2):
            drdot = 0
            i += order//2
            for j in reversed(range(i+1, order)):
                drdot += ps1.rdot[j+2]
                drdot *= h / (j-i)
            ps1.rdot[i+2] += drdot
        return ps1

    def pec(self, n, ps0, dt):
        dt = self.set_nextstep(ps0, dt)
        ps1 = self.predict(ps0.copy(), dt)
        while n > 0:
            self.evaluate(ps1)
            ps1 = self.correct(ps1, ps0, dt)
            n -= 1
        ps1 = self.interpolate(ps1, ps0, dt)
        ps1.time += dt
        ps1.nstep += 1
        return ps1


@bind_all(timings)
class H2(HX):
    """

    """
    order = 2
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
        p = self.order
        a2 = (ps.rdot[2:p+2]**2).sum(1)
        ps.tstep[...] = eta * (a2[0] / a2[1])**0.5


@bind_all(timings)
class H4(HX):
    """

    """
    order = 4
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


@bind_all(timings)
class H6(HX):
    """

    """
    order = 6
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


@bind_all(timings)
class H8(HX):
    """

    """
    order = 8
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


@bind_all(timings)
class Hermite(Base):
    """

    """
    PROVIDED_METHODS = [
        'hermite2c', 'hermite2a',
        'hermite4c', 'hermite4a',
        'hermite6c', 'hermite6a',
        'hermite8c', 'hermite8a',
    ]

    def __init__(self, ps, eta, dt_max, t_begin, method, **kwargs):
        """

        """
        super(Hermite, self).__init__(ps, eta, dt_max,
                                      t_begin, method, **kwargs)

        if method.endswith('c'):
            self.update_tstep = False
            self.shared_tstep = True
        elif method.endswith('a'):
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
