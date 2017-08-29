# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from .base import AbstractParticle, Attributes


class Blackhole(AbstractParticle):
    """

    """
    name = None
    part_type = None
    default_attr_descr = Attributes.default + [
        ('spin', '{nd}, {nb}', 'real_t', 'spin'),
        ('radius', '{nb}', 'real_t', 'radius'),
    ]
    extra_attr_descr = Attributes.extra


# -- End of File --
