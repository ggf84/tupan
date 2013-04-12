# -*- coding: utf-8 -*-
#

"""
TODO.
"""


class Base(object):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        self.eta = eta
        self.time = time
        self.particles = particles
        self.is_initialized = False

        pn_order = kwargs.pop("pn_order", 0)
        clight = kwargs.pop("clight", None)
        if pn_order > 0:
            if clight is None:
                raise TypeError(
                    "'clight' is not defined. Please set the speed of light "
                    "argument 'clight' when using 'pn_order' > 0."
                )
            else:
                from ..lib import gravity
                gravity.clight.pn_order = pn_order
                gravity.clight.clight = clight

        self.reporter = kwargs.pop("reporter", None)
        self.viewer = kwargs.pop("viewer", None)
        self.dumpper = kwargs.pop("dumpper", None)
        self.dump_freq = kwargs.pop("dump_freq", 1)
        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(type(
                self).__name__, ", ".join(kwargs.keys())))


from . import sia
from . import nreg
from . import sakura
from . import hermite


class Integrator(object):
    """

    """
    PROVIDED_METHODS = ["nreg", "sakura", "hermite", "adapthermite"]
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
        elif method == "nreg":
            logger.info("Using 'nreg' integrator.")
            self.integrator = nreg.NREG(eta, time, particles, **kwargs)
        elif method == "sakura":
            logger.info("Using 'sakura' integrator.")
            self.integrator = sakura.SAKURA(eta, time, particles, **kwargs)
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
