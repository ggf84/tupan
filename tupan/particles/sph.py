# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import Particle, typed_property
from ..lib.utils.timing import timings, bind_all


__all__ = ['Sphs']


@bind_all(timings)
class Sphs(Particle):
    """

    """
    density = typed_property('density', 'real', doc='density')


# -- End of File --
