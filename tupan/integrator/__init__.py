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

    def __new__(cls, ps, cli, *args, **kwargs):
        if cli.method in sia.SIA.PROVIDED_METHODS:
            return sia.SIA(ps, cli, *args, **kwargs)
        elif cli.method in nreg.NREG.PROVIDED_METHODS:
            return nreg.NREG(ps, cli, *args, **kwargs)
        elif cli.method in sakura.Sakura.PROVIDED_METHODS:
            return sakura.Sakura(ps, cli, *args, **kwargs)
        elif cli.method in hermite.Hermite.PROVIDED_METHODS:
            return hermite.Hermite(ps, cli, *args, **kwargs)
        else:
            msg = 'Invalid integration method: {0}.'
            raise ValueError(msg.format(cli.method))


# -- End of File --
