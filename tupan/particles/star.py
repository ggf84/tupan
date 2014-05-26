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
    spinx = typed_property('spinx', 'real')
    spiny = typed_property('spiny', 'real')
    spinz = typed_property('spinz', 'real')
    radius = typed_property('radius', 'real')
    age = typed_property('age', 'real')
    metallicity = typed_property('metallicity', 'real')

    attrs = Particle.attrs + [
        ('spinx', 'real', 'x-spin'),
        ('spiny', 'real', 'y-spin'),
        ('spinz', 'real', 'z-spin'),
        ('radius', 'real', 'radius'),
        ('age', 'real', 'age'),
        ('metallicity', 'real', 'metallicity'),
    ]


# -- End of File --
