# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from .base import AbstractParticle, AbstractNbodyMethods
from ..lib.utils.timing import timings, bind_all


@bind_all(timings)
class Sph(AbstractParticle, AbstractNbodyMethods):
    """

    """
    name = None
    part_type = None
    default_attr_descr = AbstractNbodyMethods.default_attr_descr + [
        ('density', '{n}', 'real_t', 'density'),
    ]


# -- End of File --
