# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .body import Bodies, typed_property
from ..lib.utils.timing import timings, bind_all


__all__ = ['Sphs']


@bind_all(timings)
class Sphs(Bodies):
    """

    """
    density = typed_property('density', 'real')

    attrs = Bodies.attrs + [
        ('density', 'real', 'density'),
    ]


# -- End of File --
