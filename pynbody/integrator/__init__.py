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
            self.integrator = leapfrog.LeapFrog(eta, time, particles)
            self.integrator.init_for_integration()
        elif method == "hts":
            logger.info("Using 'hts' integrator.")
            self.integrator = hts.HTS(eta, time, particles)
            self.integrator.init_for_integration()
        else:
            logger.critical("Unexpected integrator method: '%s'. Provided methods: %s",
                            method, str(self.PROVIDED_METHODS))


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
