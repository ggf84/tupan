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

    def __init__(self, *args, **kwargs):
        method = args[-1]

        self.integrator = None
        if method in sia.SIA.PROVIDED_METHODS:
            self.integrator = sia.SIA(*args, **kwargs)
        elif method in nreg.NREG.PROVIDED_METHODS:
            self.integrator = nreg.NREG(*args, **kwargs)
        elif method in sakura.Sakura.PROVIDED_METHODS:
            self.integrator = sakura.Sakura(*args, **kwargs)
        elif method in hermite.Hermite.PROVIDED_METHODS:
            self.integrator = hermite.Hermite(*args, **kwargs)
        else:
            msg = 'Invalid integration method: {0}.'
            raise ValueError(msg.format(method))

    def finalize(self, t_end):
        self.integrator.finalize(t_end)

    def evolve_step(self, t_end):
        self.integrator.evolve_step(t_end)

    @property
    def time(self):
        return self.integrator.ps.time[0]


# -- End of File --
