#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from . import hts
from . import leapfrog


class Integrator(object):
    """

    """
    PROVIDED_METHODS = ['leapfrog', 'hts']

    def __init__(self, eta, time, particles, method="leapfrog", **kwargs):
        import logging
        logger = logging.getLogger(__name__)

        self.integrator = None
        if method == "leapfrog":
            logger.info("Using 'leapfrog' integrator.")
            self.integrator = leapfrog.LeapFrog(eta, time, particles, **kwargs)
        elif method == "hts":
            logger.info("Using 'hts' integrator.")
            self.integrator = hts.HTS(eta, time, particles, **kwargs)
        else:
            logger.critical("Unexpected integrator method: '%s'. Provided methods: %s",
                            method, str(self.PROVIDED_METHODS))


    def initialize_integrator(self, t_end):
        self.integrator.initialize_integrator(t_end)

    def finalize_integrator(self, t_end):
        self.integrator.finalize_integrator(t_end)

    def step(self, t_end):
        self.integrator.step(t_end)

    @property
    def tstep(self):
        return self.integrator.tstep

    @property
    def current_time(self):
        return self.integrator.time

    @property
    def particles(self):
        return self.integrator.particles


########## end of file ##########
