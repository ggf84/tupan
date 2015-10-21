# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
from abc import ABCMeta, abstractmethod
from .base import Base, power_of_two
from ..lib.utils import with_metaclass
from ..lib.utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)


class HX(with_metaclass(ABCMeta, object)):
    """

    """
    @property
    @abstractmethod
    def order(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def prepare(ps, eta):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def predict(ps, dt):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ecorrect(ps1, ps0, dt):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def set_nextstep(ps, eta):
        raise NotImplementedError

    def __init__(self, manager):
        self.initialized = False
        self.manager = manager

    def pec(self, n, ps, eta, dt_max):
        if not self.initialized:
            self.initialized = True
            ps = self.prepare(ps, eta)
            if self.order > 4:
                n += 1
        dt = power_of_two(ps, dt_max) if self.manager.update_tstep else dt_max
        ps0 = ps.copy()
        ps1 = self.predict(ps, dt)
        while n > 0:
            ps1 = self.ecorrect(ps1, ps0, dt)
            n -= 1
        type(ps1).t_curr += dt
        ps1.time += dt
        ps1.nstep += 1
        if self.manager.update_tstep:
            self.set_nextstep(ps1, eta)
        return ps1


@bind_all(timings)
class H2(HX):
    """

    """
    order = 2

    @staticmethod
    def prepare(ps, eta):
        ps.set_tstep(ps, eta)
        ps.set_acc(ps)
        ps.jrk = ps.vel * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt

        ps.pos += h * (ps.vel)

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2

        ps1.set_acc(ps1)

        acc_p = (ps0.acc + ps1.acc)

        ps1.vel[...] = ps0.vel + h2 * (acc_p)

        vel_p = (ps0.vel + ps1.vel)

        ps1.pos[...] = ps0.pos + h2 * (vel_p)

        hinv = 1 / h

        acc_m = (ps0.acc - ps1.acc)

        jrk = -hinv * (acc_m)

        ps1.jrk[...] = jrk

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.acc**2).sum(0)
        s1 = (ps.jrk**2).sum(0)

        u = s0
        l = s1

        ps.tstep[...] = eta * (u / l)**0.5


@bind_all(timings)
class H4(HX):
    """

    """
    order = 4

    @staticmethod
    def prepare(ps, eta):
        ps.set_tstep(ps, eta)
        ps.set_acc_jrk(ps)
        ps.snp = ps.acc * 0
        ps.crk = ps.jrk * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3

        ps.pos += h * (ps.vel + h2 * (ps.acc + h3 * (ps.jrk)))
        ps.vel += h * (ps.acc + h2 * (ps.jrk))

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2
        h6 = h / 6

        ps1.set_acc_jrk(ps1)

        jrk_m = (ps0.jrk - ps1.jrk)
        acc_p = (ps0.acc + ps1.acc)

        ps1.vel[...] = ps0.vel + h2 * (acc_p + h6 * (jrk_m))

        acc_m = (ps0.acc - ps1.acc)
        vel_p = (ps0.vel + ps1.vel)

        ps1.pos[...] = ps0.pos + h2 * (vel_p + h6 * (acc_m))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2

        jrk_p = (ps0.jrk + ps1.jrk)

        snp = -hinv * (jrk_m)
        crk = 6 * hinv3 * (2 * acc_m + h * jrk_p)
        snp += h2 * (crk)

        ps1.snp[...] = snp
        ps1.crk[...] = crk

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.acc**2).sum(0)
        s1 = (ps.jrk**2).sum(0)
        s2 = (ps.snp**2).sum(0)
        s3 = (ps.crk**2).sum(0)

        u = (s0 * s2)**0.5 + s1
        l = (s1 * s3)**0.5 + s2

        ps.tstep[...] = eta * (u / l)**0.5


@bind_all(timings)
class H6(HX):
    """

    """
    order = 6

    @staticmethod
    def prepare(ps, eta):
        ps.set_tstep(ps, eta)
        ps.set_acc_jrk(ps)
        ps.set_snp_crk(ps)
        ps.crk = ps.jrk * 0
        ps.d4a = ps.snp * 0
        ps.d5a = ps.crk * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3
        h4 = h / 4
        h5 = h / 5

        ps.pos += h * (ps.vel +
                       h2 * (ps.acc +
                             h3 * (ps.jrk +
                                   h4 * (ps.snp +
                                         h5 * (ps.crk)))))
        ps.vel += h * (ps.acc + h2 * (ps.jrk + h3 * (ps.snp + h4 * (ps.crk))))
        ps.acc += h * (ps.jrk + h2 * (ps.snp + h3 * (ps.crk)))

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2
        h5 = h / 5
        h12 = h / 12

        ps1.set_snp_crk(ps1)
        ps1.set_acc_jrk(ps1)

        snp_p = (ps0.snp + ps1.snp)
        jrk_m = (ps0.jrk - ps1.jrk)
        acc_p = (ps0.acc + ps1.acc)

        ps1.vel[...] = ps0.vel + h2 * (acc_p + h5 * (jrk_m + h12 * (snp_p)))

        jrk_p = (ps0.jrk + ps1.jrk)
        acc_m = (ps0.acc - ps1.acc)
        vel_p = (ps0.vel + ps1.vel)

        ps1.pos[...] = ps0.pos + h2 * (vel_p + h5 * (acc_m + h12 * (jrk_p)))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2
        hinv5 = hinv2 * hinv3

        snp_m = (ps0.snp - ps1.snp)

        crk = 3 * hinv3 * (10 * acc_m + h * (5 * jrk_p + h2 * snp_m))
        d4a = 6 * hinv3 * (2 * jrk_m + h * snp_p)
        d5a = -60 * hinv5 * (12 * acc_m + h * (6 * jrk_p + h * snp_m))

        h4 = h / 4
        crk += h2 * (d4a + h4 * d5a)
        d4a += h2 * (d5a)

        ps1.crk[...] = crk
        ps1.d4a[...] = d4a
        ps1.d5a[...] = d5a

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.acc**2).sum(0)
        s1 = (ps.jrk**2).sum(0)
        s2 = (ps.snp**2).sum(0)
        s3 = (ps.crk**2).sum(0)
        s4 = (ps.d4a**2).sum(0)
        s5 = (ps.d5a**2).sum(0)

        u = (s0 * s2)**0.5 + s1
        l = (s3 * s5)**0.5 + s4

        ps.tstep[...] = eta * (u / l)**(1.0 / 6)


@bind_all(timings)
class H8(HX):
    """

    """
    order = 8

    @staticmethod
    def prepare(ps, eta):
        ps.set_tstep(ps, eta)
        ps.set_acc_jrk(ps)
        ps.set_snp_crk(ps)
        ps.d4a = ps.snp * 0
        ps.d5a = ps.crk * 0
        ps.d6a = ps.d4a * 0
        ps.d7a = ps.d5a * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3
        h4 = h / 4
        h5 = h / 5
        h6 = h / 6
        h7 = h / 7

        ps.pos += h * (ps.vel +
                       h2 * (ps.acc +
                             h3 * (ps.jrk +
                                   h4 * (ps.snp +
                                         h5 * (ps.crk +
                                               h6 * (ps.d4a +
                                                     h7 * (ps.d5a)))))))
        ps.vel += h * (ps.acc +
                       h2 * (ps.jrk +
                             h3 * (ps.snp +
                                   h4 * (ps.crk +
                                         h5 * (ps.d4a +
                                               h6 * (ps.d5a))))))
        ps.acc += h * (ps.jrk +
                       h2 * (ps.snp +
                             h3 * (ps.crk +
                                   h4 * (ps.d4a +
                                         h5 * (ps.d5a)))))
        ps.jrk += h * (ps.snp + h2 * (ps.crk + h3 * (ps.d4a + h4 * (ps.d5a))))

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3
        h14 = h / 14
        h20 = h / 20

        ps1.set_snp_crk(ps1)
        ps1.set_acc_jrk(ps1)

        crk_m = (ps0.crk - ps1.crk)
        snp_p = (ps0.snp + ps1.snp)
        jrk_m = (ps0.jrk - ps1.jrk)
        acc_p = (ps0.acc + ps1.acc)

        ps1.vel[...] = ps0.vel + h2 * (acc_p +
                                       h14 * (3 * jrk_m +
                                              h3 * (snp_p +
                                                    h20 * (crk_m))))

        snp_m = (ps0.snp - ps1.snp)
        jrk_p = (ps0.jrk + ps1.jrk)
        acc_m = (ps0.acc - ps1.acc)
        vel_p = (ps0.vel + ps1.vel)

        ps1.pos[...] = ps0.pos + h2 * (vel_p +
                                       h14 * (3 * acc_m +
                                              h3 * (jrk_p +
                                                    h20 * (snp_m))))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2
        hinv5 = hinv2 * hinv3
        hinv7 = hinv2 * hinv5

        crk_p = (ps0.crk + ps1.crk)

        d4a = 3 * hinv3 * (10 * jrk_m + h * (5 * snp_p + h2 * crk_m))
        d5a = -15 * hinv5 * (168 * acc_m +
                             h * (84 * jrk_p +
                                  h * (16 * snp_m +
                                       h * crk_p)))
        d6a = -60 * hinv5 * (12 * jrk_m + h * (6 * snp_p + h * crk_m))
        d7a = 840 * hinv7 * (120 * acc_m +
                             h * (60 * jrk_p +
                                  h * (12 * snp_m +
                                       h * crk_p)))

        h4 = h / 4
        h6 = h / 6
        d4a += h2 * (d5a + h4 * (d6a + h6 * d7a))
        d5a += h2 * (d6a + h4 * d7a)
        d6a += h2 * (d7a)

        ps1.d4a[...] = d4a
        ps1.d5a[...] = d5a
        ps1.d6a[...] = d6a
        ps1.d7a[...] = d7a

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.acc**2).sum(0)
        s1 = (ps.jrk**2).sum(0)
        s2 = (ps.snp**2).sum(0)
#        s3 = (ps.crk**2).sum(0)
#        s4 = (ps.d4a**2).sum(0)
        s5 = (ps.d5a**2).sum(0)
        s6 = (ps.d6a**2).sum(0)
        s7 = (ps.d7a**2).sum(0)

        u = (s0 * s2)**0.5 + s1
        l = (s5 * s7)**0.5 + s6

        ps.tstep[...] = eta * (u / l)**(1.0 / 10)


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

    def do_step(self, ps, dt_max):
        """

        """
        return self.hermite.pec(2, ps, self.eta, dt_max)


# -- End of File --
