# -*- coding: utf-8 -*-
#

"""

"""

import pickle
import logging
import numpy as np
from itertools import count


def power_of_two(ps, dt_max):
    """

    """
    power = int(np.log2(ps.tstep_min) - 1)
    dtq = 2.0**power
    dtq = min(dtq, abs(dt_max))

    time = ps.global_time
    while time % dtq != 0:
        dtq /= 2

    dtq = np.copysign(dtq, dt_max)
    return dtq


class Base(object):
    """

    """
    PROVIDED_METHODS = None

    def __init__(self, ps, cli, viewer=None, dumpper=None, checker=None):
        self.ps = ps
        self.cli = cli
        self.viewer = viewer
        self.dumpper = dumpper
        self.checker = checker

        self.step = count(0)
        self.t_next = cli.t_begin + cli.dt_max

    def __enter__(self):
        LOGGER = logging.getLogger(self.__module__)
        LOGGER.debug(type(self).__name__+'.__enter__')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        LOGGER = logging.getLogger(self.__module__)
        LOGGER.debug(type(self).__name__+'.__exit__')
        with open('restart.pkl', 'wb') as fid:
            protocol = pickle.HIGHEST_PROTOCOL
            pickle.dump(self, fid, protocol=protocol)

    def __getstate__(self):  # viewer can't be pickled!
        dct = vars(self).copy()
        dct['viewer'] = None
        return dct

    def do_step(self, ps, dt):
        """

        """
        raise NotImplementedError

    def evolve_step(self, dt, step):
        """

        """
        while abs(self.time) < self.t_next:
            self.ps = self.do_step(self.ps, dt)
        self.t_next += self.cli.dt_max

        if self.viewer:
            self.viewer.show_event(self.ps)
        if self.checker:
            self.checker.print_diagnostic(self.ps)
        if self.dumpper:
            with self.dumpper as io:
                io.flush_stream(tag=step)
                io.dump_snap(self.ps, tag=step+1)

    def evolve(self, t_end):
        """

        """
        dt = self.cli.dt_max
        if not self.update_tstep:
            dt *= self.cli.eta

        if self.viewer:
            self.viewer.show_event(self.ps)

        while abs(self.time) < t_end:
            self.evolve_step(dt, next(self.step))

    @property
    def time(self):
        return self.ps.global_time


# -- End of File --
