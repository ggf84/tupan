#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO.
"""


from . import sia
from . import bios
from . import nreg
from . import hermite


class Integrator(object):
    """

    """
    PROVIDED_METHODS = ["bios", "nreg", "hermite", "adapthermite"]
    PROVIDED_METHODS.extend(sia.SIA.PROVIDED_METHODS)

    def __init__(self, eta, time, particles, **kwargs):
        import logging
        logger = logging.getLogger(__name__)

        method = kwargs.pop("method", None)

        self.integrator = None
        if method in sia.SIA.PROVIDED_METHODS:
            logger.info("Using %s integrator.", method)
            self.integrator = sia.SIA(eta, time, particles, method, **kwargs)
        elif method == "hermite":
            logger.info("Using 'hermite' integrator.")
            self.integrator = hermite.Hermite(eta, time, particles, **kwargs)
        elif method == "adapthermite":
            logger.info("Using 'adapthermite' integrator.")
            self.integrator = hermite.AdaptHermite(
                eta, time, particles, **kwargs)
        elif method == "bios":
            logger.info("Using 'bios' integrator.")
            self.integrator = bios.BIOS(eta, time, particles, **kwargs)
        elif method == "nreg":
            logger.info("Using 'nreg' integrator.")
            self.integrator = nreg.NREG(eta, time, particles, **kwargs)
        else:
            logger.critical(
                "Unexpected integrator method: '%s'. Provided methods: %s",
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
