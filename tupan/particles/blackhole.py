# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import AbstractParticle, AbstractNbodyMethods
from ..lib.utils.timing import timings, bind_all


__all__ = ['Blackhole']


@bind_all(timings)
class Blackhole(AbstractParticle, AbstractNbodyMethods):
    """

    """
    name = None
    default_attr_descr = AbstractNbodyMethods.default_attr_descr + [
        ('spin', (3,), 'real', 'spin'),
        ('radius', (), 'real', 'radius'),
    ]


# -- End of File --
