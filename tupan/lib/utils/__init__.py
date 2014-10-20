# -*- coding: utf-8 -*-
#

"""
TODO.
"""


# function inspired by package six.
def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):
        def __new__(cls, name, this_bases=None, d=None):
            return meta(name, bases, d or {})
    return type.__new__(metaclass, 'temporary_class', (), {})


# -- End of File --
