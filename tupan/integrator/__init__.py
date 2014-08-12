# -*- coding: utf-8 -*-
#

"""
TODO.
"""


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

        method = kwargs.pop('method', None)

        self.integrator = None
        if method in sia.SIA.PROVIDED_METHODS:
            logger.info('Using %s integrator.', method)
            self.integrator = sia.SIA(eta, time, ps, method, **kwargs)
        elif method in nreg.NREG.PROVIDED_METHODS:
            logger.info('Using %s integrator.', method)
            self.integrator = nreg.NREG(eta, time, ps, method, **kwargs)
        elif method in sakura.Sakura.PROVIDED_METHODS:
            logger.info('Using %s integrator.', method)
            self.integrator = sakura.Sakura(eta, time, ps, method, **kwargs)
        elif method in hermite.Hermite.PROVIDED_METHODS:
            logger.info('Using %s integrator.', method)
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
