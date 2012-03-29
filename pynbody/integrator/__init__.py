#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from . import block
from . import leapfrog


class Integrator(object):
    """

    """
    PROVIDED_METHODS = ['leapfrog', 'blockstep']

    def __init__(self, eta, current_time, particles, method_name="leapfrog", **kwargs):
        import logging
        logger = logging.getLogger(__name__)

        self.integrator = None
        if method_name == "leapfrog":
            logger.info("Using 'leapfrog' integrator.")
            self.integrator = leapfrog.LeapFrog(eta, current_time, particles)
        elif method_name == "blockstep":
            logger.info("Using 'blockstep' integrator.")
            self.integrator = leapfrog.BlockStep(eta, current_time, particles)
        else:
            logger.critical("Unexpected integrator name: '%s'. Provided methods: %s", method_name, str(self.PROVIDED_METHODS))


    def step(self):
        self.integrator.step()

    @property
    def tstep(self):
        return self.integrator.tstep

    @property
    def current_time(self):
        return self.integrator.current_time

    @property
    def particles(self):
        return self.integrator.particles


########## end of file ##########
