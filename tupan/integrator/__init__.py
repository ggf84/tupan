# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import math
import numpy as np


class Base(object):
    """

    """
    def __init__(self, eta, time, ps, **kwargs):
        self.eta = eta
        type(ps).t_curr = time
        self.ps = ps
        self.is_initialized = False

        pn_order = kwargs.pop("pn_order", 0)
        clight = kwargs.pop("clight", None)
        type(ps).include_pn_corrections = False
        if pn_order > 0:
            if clight is None:
                raise TypeError(
                    "'clight' is not defined. Please set the speed of light "
                    "argument 'clight' when using 'pn_order' > 0."
                )
            else:
                from ..lib import extensions
                extensions.clight.pn_order = pn_order
                extensions.clight.clight = clight
                type(ps).include_pn_corrections = True

        self.reporter = kwargs.pop("reporter", None)
        self.viewer = kwargs.pop("viewer", None)
        self.dumpper = kwargs.pop("dumpper", None)
        self.dump_freq = kwargs.pop("dump_freq", 1)
        self.gl_freq = kwargs.pop("gl_freq", 1)
        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(type(
                self).__name__, ", ".join(kwargs.keys())))

    def get_base_tstep(self, t_end):
        """

        """
        import sys
        ps = self.ps
        t_curr = ps.t_curr
        dt = min(abs(t_end)-abs(t_curr), abs(self.eta))
        dt = max(dt, abs(t_end)*(2*sys.float_info.epsilon))
        return math.copysign(dt, self.eta)

    def get_min_block_tstep(self, ps, tau):
        """

        """
        min_ts = ps.min_tstep()

        power = int(np.log2(min_ts) - 1)
        min_bts = 2.0**power

        t_curr = ps.t_curr
        t_next = t_curr + min_bts
        while t_next % min_bts != 0:
            min_bts /= 2

        min_bts = math.copysign(min_bts, tau)

        if abs(min_bts) > abs(tau):
            min_bts = tau

        return min_bts

    def initialize(self, t_end):
        """

        """
        raise NotImplementedError

    def finalize(self, t_end):
        """

        """
        raise NotImplementedError

    def do_step(self, ps, tau):
        """

        """
        raise NotImplementedError

    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        ps = self.ps

        self.wl = type(ps)()
        tau = self.get_base_tstep(t_end)

        ps = self.do_step(ps, tau)

        if self.reporter:
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.dump_worldline(self.wl)

        self.ps = ps


from . import sia
from . import nreg
from . import sakura
from . import hermite


class Integrator(object):
    """

    """
    PROVIDED_METHODS = []
    PROVIDED_METHODS.extend(sia.SIA.PROVIDED_METHODS)
    PROVIDED_METHODS.extend(nreg.NREG.PROVIDED_METHODS)
    PROVIDED_METHODS.extend(sakura.Sakura.PROVIDED_METHODS)
    PROVIDED_METHODS.extend(hermite.Hermite.PROVIDED_METHODS)

    def __init__(self, eta, time, ps, **kwargs):
        import logging
        logger = logging.getLogger(__name__)

        method = kwargs.pop("method", None)

        self.integrator = None
        if method in sia.SIA.PROVIDED_METHODS:
            logger.info("Using %s integrator.", method)
            self.integrator = sia.SIA(eta, time, ps, method, **kwargs)
        elif method in nreg.NREG.PROVIDED_METHODS:
            logger.info("Using %s integrator.", method)
            self.integrator = nreg.NREG(eta, time, ps, method, **kwargs)
        elif method in sakura.Sakura.PROVIDED_METHODS:
            logger.info("Using %s integrator.", method)
            self.integrator = sakura.Sakura(eta, time, ps, method, **kwargs)
        elif method in hermite.Hermite.PROVIDED_METHODS:
            logger.info("Using %s integrator.", method)
            self.integrator = hermite.Hermite(eta, time, ps, method, **kwargs)
        else:
            logger.critical(
                "Unexpected integration method: '%s'. Provided methods: %s",
                method, str(self.PROVIDED_METHODS))

    def initialize(self, t_end):
        self.integrator.initialize(t_end)

    def finalize(self, t_end):
        self.integrator.finalize(t_end)

    def evolve_step(self, t_end):
        self.integrator.evolve_step(t_end)

    @property
    def time(self):
        return self.integrator.ps.t_curr

    @property
    def particle_system(self):
        return self.integrator.ps


# -- End of File --
