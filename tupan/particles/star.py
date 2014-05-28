# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import Particle, typed_property
from ..lib.utils.timing import timings, bind_all


__all__ = ['Stars']


@bind_all(timings)
class Stars(Particle):
    """

    """
    dtype = []

    spinx = typed_property('spinx', 'real', doc='x-spin')
    spiny = typed_property('spiny', 'real', doc='y-spin')
    spinz = typed_property('spinz', 'real', doc='z-spin')
    radius = typed_property('radius', 'real', doc='radius')
    age = typed_property('age', 'real', doc='age')
    metallicity = typed_property('metallicity', 'real', doc='metallicity')


# -- End of File --
