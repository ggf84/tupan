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
        ps.rdot[3] = ps.rdot[1] * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt

        ps.rdot[0] += h * (ps.rdot[1])

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2

        ps1.set_acc(ps1)

        acc_p = (ps0.rdot[2] + ps1.rdot[2])

        ps1.rdot[1][...] = ps0.rdot[1] + h2 * (acc_p)

        vel_p = (ps0.rdot[1] + ps1.rdot[1])

        ps1.rdot[0][...] = ps0.rdot[0] + h2 * (vel_p)

        hinv = 1 / h

        acc_m = (ps0.rdot[2] - ps1.rdot[2])

        jrk = -hinv * (acc_m)

        ps1.rdot[3][...] = jrk

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.rdot[2]**2).sum(0)
        s1 = (ps.rdot[3]**2).sum(0)

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
        ps.rdot[4] = ps.rdot[2] * 0
        ps.rdot[5] = ps.rdot[3] * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3

        ps.rdot[0] += h * (ps.rdot[1] + h2 * (ps.rdot[2] + h3 * (ps.rdot[3])))
        ps.rdot[1] += h * (ps.rdot[2] + h2 * (ps.rdot[3]))

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2
        h6 = h / 6

        ps1.set_acc_jrk(ps1)

        jrk_m = (ps0.rdot[3] - ps1.rdot[3])
        acc_p = (ps0.rdot[2] + ps1.rdot[2])

        ps1.rdot[1][...] = ps0.rdot[1] + h2 * (acc_p + h6 * (jrk_m))

        acc_m = (ps0.rdot[2] - ps1.rdot[2])
        vel_p = (ps0.rdot[1] + ps1.rdot[1])

        ps1.rdot[0][...] = ps0.rdot[0] + h2 * (vel_p + h6 * (acc_m))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2

        jrk_p = (ps0.rdot[3] + ps1.rdot[3])

        snp = -hinv * (jrk_m)
        crk = 6 * hinv3 * (2 * acc_m + h * jrk_p)
        snp += h2 * (crk)

        ps1.rdot[4][...] = snp
        ps1.rdot[5][...] = crk

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.rdot[2]**2).sum(0)
        s1 = (ps.rdot[3]**2).sum(0)
        s2 = (ps.rdot[4]**2).sum(0)
        s3 = (ps.rdot[5]**2).sum(0)

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
        ps.rdot[5] = ps.rdot[3] * 0
        ps.rdot[6] = ps.rdot[4] * 0
        ps.rdot[7] = ps.rdot[5] * 0
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

        ps.rdot[0] += h * (ps.rdot[1] +
                           h2 * (ps.rdot[2] +
                                 h3 * (ps.rdot[3] +
                                       h4 * (ps.rdot[4] +
                                             h5 * (ps.rdot[5])))))
        ps.rdot[1] += h * (ps.rdot[2] +
                           h2 * (ps.rdot[3] +
                                 h3 * (ps.rdot[4] +
                                       h4 * (ps.rdot[5]))))
        ps.rdot[2] += h * (ps.rdot[3] + h2 * (ps.rdot[4] + h3 * (ps.rdot[5])))

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

        snp_p = (ps0.rdot[4] + ps1.rdot[4])
        jrk_m = (ps0.rdot[3] - ps1.rdot[3])
        acc_p = (ps0.rdot[2] + ps1.rdot[2])

        ps1.rdot[1][...] = ps0.rdot[1] + h2 * (acc_p +
                                               h5 * (jrk_m +
                                                     h12 * (snp_p)))

        jrk_p = (ps0.rdot[3] + ps1.rdot[3])
        acc_m = (ps0.rdot[2] - ps1.rdot[2])
        vel_p = (ps0.rdot[1] + ps1.rdot[1])

        ps1.rdot[0][...] = ps0.rdot[0] + h2 * (vel_p +
                                               h5 * (acc_m +
                                                     h12 * (jrk_p)))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2
        hinv5 = hinv2 * hinv3

        snp_m = (ps0.rdot[4] - ps1.rdot[4])

        crk = 3 * hinv3 * (10 * acc_m + h * (5 * jrk_p + h2 * snp_m))
        d4a = 6 * hinv3 * (2 * jrk_m + h * snp_p)
        d5a = -60 * hinv5 * (12 * acc_m + h * (6 * jrk_p + h * snp_m))

        h4 = h / 4
        crk += h2 * (d4a + h4 * d5a)
        d4a += h2 * (d5a)

        ps1.rdot[5][...] = crk
        ps1.rdot[6][...] = d4a
        ps1.rdot[7][...] = d5a

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.rdot[2]**2).sum(0)
        s1 = (ps.rdot[3]**2).sum(0)
        s2 = (ps.rdot[4]**2).sum(0)
        s3 = (ps.rdot[5]**2).sum(0)
        s4 = (ps.rdot[6]**2).sum(0)
        s5 = (ps.rdot[7]**2).sum(0)

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
        ps.rdot[6] = ps.rdot[4] * 0
        ps.rdot[7] = ps.rdot[5] * 0
        ps.rdot[8] = ps.rdot[6] * 0
        ps.rdot[9] = ps.rdot[7] * 0
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

        ps.rdot[0] += h * (ps.rdot[1] +
                           h2 * (ps.rdot[2] +
                                 h3 * (ps.rdot[3] +
                                       h4 * (ps.rdot[4] +
                                             h5 * (ps.rdot[5] +
                                                   h6 * (ps.rdot[6] +
                                                         h7 * (ps.rdot[7])))))))
        ps.rdot[1] += h * (ps.rdot[2] +
                           h2 * (ps.rdot[3] +
                                 h3 * (ps.rdot[4] +
                                       h4 * (ps.rdot[5] +
                                             h5 * (ps.rdot[6] +
                                                   h6 * (ps.rdot[7]))))))
        ps.rdot[2] += h * (ps.rdot[3] +
                           h2 * (ps.rdot[4] +
                                 h3 * (ps.rdot[5] +
                                       h4 * (ps.rdot[6] +
                                             h5 * (ps.rdot[7])))))
        ps.rdot[3] += h * (ps.rdot[4] +
                           h2 * (ps.rdot[5] +
                                 h3 * (ps.rdot[6] +
                                       h4 * (ps.rdot[7]))))

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

        crk_m = (ps0.rdot[5] - ps1.rdot[5])
        snp_p = (ps0.rdot[4] + ps1.rdot[4])
        jrk_m = (ps0.rdot[3] - ps1.rdot[3])
        acc_p = (ps0.rdot[2] + ps1.rdot[2])

        ps1.rdot[1][...] = ps0.rdot[1] + h2 * (acc_p +
                                               h14 * (3 * jrk_m +
                                                      h3 * (snp_p +
                                                            h20 * (crk_m))))

        snp_m = (ps0.rdot[4] - ps1.rdot[4])
        jrk_p = (ps0.rdot[3] + ps1.rdot[3])
        acc_m = (ps0.rdot[2] - ps1.rdot[2])
        vel_p = (ps0.rdot[1] + ps1.rdot[1])

        ps1.rdot[0][...] = ps0.rdot[0] + h2 * (vel_p +
                                               h14 * (3 * acc_m +
                                                      h3 * (jrk_p +
                                                            h20 * (snp_m))))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2
        hinv5 = hinv2 * hinv3
        hinv7 = hinv2 * hinv5

        crk_p = (ps0.rdot[5] + ps1.rdot[5])

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

        ps1.rdot[6][...] = d4a
        ps1.rdot[7][...] = d5a
        ps1.rdot[8][...] = d6a
        ps1.rdot[9][...] = d7a

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.rdot[2]**2).sum(0)
        s1 = (ps.rdot[3]**2).sum(0)
        s2 = (ps.rdot[4]**2).sum(0)
#        s3 = (ps.rdot[5]**2).sum(0)
#        s4 = (ps.rdot[6]**2).sum(0)
        s5 = (ps.rdot[7]**2).sum(0)
        s6 = (ps.rdot[8]**2).sum(0)
        s7 = (ps.rdot[9]**2).sum(0)

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
