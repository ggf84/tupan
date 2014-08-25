# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import Particle
from ..lib.utils.timing import timings, bind_all


__all__ = ['Blackholes']


@bind_all(timings)
class Blackholes(Particle):
    """

    """
    name = None
    default_attr_descr = Particle.default_attr_descr + [
        ('spin', (3,), 'real', 'spin'),
        ('radius', (), 'real', 'radius'),
    ]


# -- End of File --
