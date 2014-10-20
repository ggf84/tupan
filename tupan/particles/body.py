# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import AbstractParticle, AbstractNbodyMethods
from ..lib.utils.timing import timings, bind_all


__all__ = ['Bodies']


@bind_all(timings)
class Bodies(AbstractParticle, AbstractNbodyMethods):
    """

    """
    name = None
    default_attr_descr = AbstractNbodyMethods.default_attr_descr + []


# -- End of File --
