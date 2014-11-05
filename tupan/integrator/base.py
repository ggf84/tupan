# -*- coding: utf-8 -*-
#

"""

"""


import sys
import math
import numpy as np


def power_of_two(ps, dtmax):
    """

    """
    tcurr = ps.t_curr
    dt = abs(ps.tstep).min()

    power = int(np.log2(abs(dt)) - 1)
    dtq = 2.0**power
    dtq = min(dtq, abs(dtmax))
    dtq = math.copysign(dtq, dtmax)

    tnext = tcurr + dtq
    while tnext % dtq != 0:
        dtq /= 2

    return dtq


class Base(object):
    """

    """
    def __init__(self, eta, time, ps, **kwargs):
        self.eta = eta
        type(ps).t_curr = time
        self.ps = ps
        self.is_initialized = False

        pn_order = kwargs.pop('pn_order', 0)
        clight = kwargs.pop('clight', None)
        if pn_order > 0:
            if clight is None:
                raise TypeError(
                    "'clight' is not defined. Please set the speed of light "
                    "argument 'clight' when using 'pn_order' > 0."
                )
            else:
                from ..lib import extensions
                extensions.pn.order = pn_order
                extensions.pn.clight = clight
                type(self.ps).include_pn_corrections = True

        self.reporter = kwargs.pop('reporter', None)
        self.viewer = kwargs.pop('viewer', None)
        self.dumpper = kwargs.pop('dumpper', None)
        self.dump_freq = kwargs.pop('dump_freq', 1)
        self.gl_freq = kwargs.pop('gl_freq', 1)
        if kwargs:
            msg = '{0}.__init__ received unexpected keyword arguments: {1}.'
            raise TypeError(msg.format(type(
                self).__name__, ', '.join(kwargs.keys())))

    def __getstate__(self):  # apparently vispy objects can't be pickled!
        dct = vars(self).copy()
        del dct['viewer']
        return dct

    def get_base_tstep(self, t_end):
        """

        """
        ps = self.ps
        t_curr = ps.t_curr
        dt = min(abs(t_end)-abs(t_curr), abs(self.eta))
        dt = max(dt, abs(t_end)*(2*sys.float_info.epsilon))
        return math.copysign(dt, self.eta)

    def initialize(self, t_end):
        """

        """
        raise NotImplementedError

    def finalize(self, t_end):
        """

        """
        raise NotImplementedError

    def do_step(self, ps, dt):
        """

        """
        raise NotImplementedError

    def dump(self, dt, ps):
        if self.dumpper:
            slc = abs(ps.time // dt) % self.dump_freq == 0
            if any(slc):
                self.dumpper.wl.append(ps[slc])
        if self.viewer:
            slc = abs(ps.time // dt) % self.gl_freq == 0
            if any(slc):
                self.viewer.show_event(ps[slc])

    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        ps = self.ps

        dt = self.get_base_tstep(t_end)

        ps = self.do_step(ps, dt)

        if self.reporter:
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.flush_worldline()

        self.ps = ps


# -- End of File --
