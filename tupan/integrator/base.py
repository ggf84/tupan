# -*- coding: utf-8 -*-
#

"""

"""

import math
import logging
import numpy as np


LOGGER = logging.getLogger(__name__)


def power_of_two(ps, dt_max):
    """

    """
    dt = abs(ps.tstep).min()

    power = int(np.log2(dt) - 1)
    dtq = 2.0**power
    dtq = min(dtq, abs(dt_max))
    dtq = np.copysign(dtq, dt_max)

    while ps.time[0] % dtq != 0:
        dtq /= 2

    return dtq


class Base(object):
    """

    """
    PROVIDED_METHODS = None

    def __getstate__(self):  # apparently vispy objects can't be pickled!
        dct = vars(self).copy()
        del dct['viewer']
        return dct

    def __init__(self, ps, eta, dt_max, t_begin, method, **kwargs):
        if method not in self.PROVIDED_METHODS:
            raise ValueError('Invalid integration method: {0}'.format(method))

        LOGGER.info("Initializing '%s' integrator at "
                    "t_begin = %g.", method, t_begin)

        self.ps = ps
        self.eta = eta
        self.dt_max = math.copysign(dt_max, eta)
        ps.time[...] = t_begin
        type(ps).t_next = t_begin
        self.method = method

        self.pn = kwargs.pop('pn', None)
        self.viewer = kwargs.pop('viewer', None)
        self.dumpper = kwargs.pop('dumpper', None)
        self.reporter = kwargs.pop('reporter', None)
        self.dump_freq = kwargs.pop('dump_freq', 1)
        if kwargs:
            msg = '{0}.__init__ received unexpected keyword arguments: {1}.'
            raise TypeError(msg.format(type(
                self).__name__, ', '.join(kwargs.keys())))

        if self.reporter:
            self.reporter.init_diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.init_new_era(ps)
        if self.viewer:
            self.viewer.show_event(ps)

    def do_step(self, ps, dt):
        """

        """
        raise NotImplementedError

    def evolve_step(self, t_end):
        """

        """
        eta = self.eta
        dt = self.dt_max
        if not self.update_tstep:
            dt = min(abs(dt), abs(eta))
            dt = math.copysign(dt, eta)

        type(self.ps).t_next += self.dt_max
        while self.ps.t_next > self.ps.time[0]:
            self.ps = self.do_step(self.ps, dt)

        if self.reporter:
            self.reporter.diagnostic_report(self.ps)
        if self.dumpper:
            self.dumpper.init_new_era(self.ps)
        if self.viewer:
            self.viewer.show_event(self.ps)

    def finalize(self, t_end):
        """

        """
        LOGGER.info("Finalizing '%s' integrator at "
                    "t_end = %g.", self.method, t_end)

        if self.viewer:
            self.viewer.show_event(self.ps)
            self.viewer.enter_main_loop()


# -- End of File --
