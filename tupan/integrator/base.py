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
    tcurr = ps.t_curr
    dt = abs(ps.tstep).min()

    power = int(np.log2(abs(dt)) - 1)
    dtq = 2.0**power
    dtq = min(dtq, abs(dt_max))
    dtq = np.copysign(dtq, dt_max)

    tnext = tcurr + dtq
    while tnext % dtq != 0:
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
        type(ps).t_curr = t_begin
        type(ps).t_next = t_begin
        self.method = method

        pn_order = kwargs.pop('pn_order', 0)
        clight = kwargs.pop('clight', float('inf'))
        if pn_order > 0:
            if clight == float('inf'):
                raise ValueError(
                    "By default 'clight' == float('inf'). Please set "
                    "the speed of light argument 'clight' to a finite "
                    "value if you chose 'pn_order' > 0."
                )
            else:
                from ..lib import extensions
                extensions.pn = extensions.PN(pn_order, clight)
                type(self.ps).include_pn_corrections = True
                for member in self.ps.members.values():
                    type(member).include_pn_corrections = True

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
        dt_max = self.dt_max
        if not self.update_tstep:
            dt_max = min(abs(dt_max), abs(eta))
            dt_max = math.copysign(dt_max, eta)

        type(self.ps).t_next += self.dt_max
        while self.ps.t_next > self.ps.t_curr:
            self.ps = self.do_step(self.ps, dt_max)

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
