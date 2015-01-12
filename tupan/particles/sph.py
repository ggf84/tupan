# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import AbstractParticle, AbstractNbodyMethods
from ..lib.utils.timing import timings, bind_all


__all__ = ['Sph']


@bind_all(timings)
class Sph(AbstractParticle, AbstractNbodyMethods):
    """

    """
    name = None
    default_attr_descr = AbstractNbodyMethods.default_attr_descr + [
        ('density', '{n}', 'real', 'density'),
    ]


# -- End of File --
