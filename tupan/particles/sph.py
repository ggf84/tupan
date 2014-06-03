# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import Particle
from ..lib.utils.timing import timings, bind_all


__all__ = ['Sphs']


@bind_all(timings)
class Sphs(Particle):
    """

    """
    dtype = None
    attr_descr = Particle.attr_descr + [
        ('density', 'real', 'density'), ]


# -- End of File --
