# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import Particle
from ..lib.utils.timing import timings, bind_all


__all__ = ['Stars']


@bind_all(timings)
class Stars(Particle):
    """

    """
    dtype = None
    attr_descr = Particle.attr_descr + [
        ('spin', (3,), 'real', 'spin'),
        ('radius', (), 'real', 'radius'),
        ('age', (), 'real', 'age'),
        ('metallicity', (), 'real', 'metallicity'), ]


# -- End of File --
