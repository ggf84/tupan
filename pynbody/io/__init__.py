#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from . import hdf5io


class IO(object):
    """

    """
    __INTERFACES__ = [hdf5io.HDF5IO,
                     ]
    INTERFACES = dict([(i.__name__.lower(), i) for i in __INTERFACES__])
    del i
    del __INTERFACES__

    def __init__(self, fname, **kwargs):
        fmode = kwargs.pop("fmode", 'a')
        interface = kwargs.pop("interface", 'hdf5io')
        if not interface in self.INTERFACES:
            msg = "{0}.__init__ received unexpected interface name: '{1}'."
            raise TypeError(msg.format(self.__class__.__name__, interface))

        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        Interface = self.INTERFACES[interface]
        self._io = Interface(fname, fmode)


    def write_snapshot(self, *args, **kwargs):
        self._io.write_snapshot(*args, **kwargs)


    def read_snapshot(self, *args, **kwargs):
        return self._io.read_snapshot(*args, **kwargs)


########## end of file ##########
