#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from . import hts
from . import hermite
from . import leapfrog


class Integrator(object):
    """

    """
    PROVIDED_METHODS = ['leapfrog', 'adaptlf', 'hts', 'hermite']

    def __init__(self, eta, time, particles, **kwargs):
        import logging
        logger = logging.getLogger(__name__)

        method = kwargs.pop("method", None)

        self.integrator = None
        if method == "leapfrog":
            logger.info("Using 'leapfrog' integrator.")
            self.integrator = leapfrog.LeapFrog(eta, time, particles, **kwargs)
        elif method == "adaptlf":
            logger.info("Using 'adaptlf' integrator.")
            self.integrator = leapfrog.AdaptLF(eta, time, particles, **kwargs)
        elif method == "hts":
            logger.info("Using 'hts' integrator.")
            self.integrator = hts.HTS(eta, time, particles, **kwargs)
        elif method == "hermite":
            logger.info("Using 'hermite' integrator.")
            self.integrator = hermite.Hermite(eta, time, particles, **kwargs)
        else:
            logger.critical("Unexpected integrator method: '%s'. Provided methods: %s",
                            method, str(self.PROVIDED_METHODS))


    def initialize(self, t_end):
        self.integrator.initialize(t_end)

    def finalize(self, t_end):
        self.integrator.finalize(t_end)

    def evolve_step(self, t_end):
        self.integrator.evolve_step(t_end)

    @property
    def tstep(self):
        return self.integrator.tstep

    @property
    def time(self):
        return self.integrator.time

    @property
    def particles(self):
        return self.integrator.particles


########## end of file ##########
