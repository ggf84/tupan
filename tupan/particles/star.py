# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from .base import AbstractParticle, Attributes


class Star(AbstractParticle):
    """

    """
    name = None
    part_type = None
    default_attr_descr = Attributes.default + [
        ('age', '{nb}', 'real_t', 'age'),
        ('spin', '{nd}, {nb}', 'real_t', 'spin'),
        ('radius', '{nb}', 'real_t', 'radius'),
        ('metallicity', '{nb}', 'real_t', 'metallicity'),
    ]
    extra_attr_descr = Attributes.extra


# -- End of File --
