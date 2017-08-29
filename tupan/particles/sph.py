# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from .base import AbstractParticle, Attributes


class Sph(AbstractParticle):
    """

    """
    name = None
    part_type = None
    default_attr_descr = Attributes.default + [
        ('density', '{nb}', 'real_t', 'density at particle position'),
        ('pressure', '{nb}', 'real_t', 'pressure at particle position'),
        ('viscosity', '{nb}', 'real_t', 'viscosity at particle position'),
        ('temperature', '{nb}', 'real_t', 'temperature at particle position'),
    ]
    extra_attr_descr = Attributes.extra


# -- End of File --
