#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from . import block
from . import leapfrog


class Integrator(object):
    """

    """
    __METHS__ = [leapfrog.LeapFrog,
                 block.BlockStep,
                ]
    METHS = dict([(m.__name__.lower(), m) for m in __METHS__])
    del m
    del __METHS__

    def __init__(self, eta, time, particles, **kwargs):
        meth_name = kwargs.pop("meth", "leapfrog")
        if not meth_name in self.METHS:
            msg = "{0}.__init__ received unexpected meth name: '{1}'."
            raise TypeError(msg.format(self.__class__.__name__, meth_name))

        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        Meth = self.METHS[meth_name]
        self._meth = Meth(eta, time, particles)

    def step(self):
        self._meth.step()

    @property
    def tstep(self):
        return self._meth.tstep

    @property
    def time(self):
        return self._meth.time

    @property
    def particles(self):
        return self._meth.particles


########## end of file ##########
