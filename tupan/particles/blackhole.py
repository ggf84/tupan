# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import Particle, typed_property
from ..lib.utils.timing import timings, bind_all


__all__ = ['Blackholes']


@bind_all(timings)
class Blackholes(Particle):
    """

    """
    spinx = typed_property('spinx', 'real', doc='x-spin')
    spiny = typed_property('spiny', 'real', doc='y-spin')
    spinz = typed_property('spinz', 'real', doc='z-spin')
    radius = typed_property('radius', 'real', doc='radius')


# -- End of File --
