#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np



class Pbase(object):
    """
    A base class implementing common functionalities for all types of particles.
    """

    def __init__(self, dtype):
        self._data = np.zeros(10, dtype=dtype)
        for attr in self._data.dtype.names:
            item = self._data[attr]
            setattr(self, attr, item)

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self._data)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __reversed__(self):
        return reversed(self._data)


    def __getitem__(self, index):
        if not isinstance(index, slice):
            index = slice(index, index+1)
        s = self._data[index].copy()
        ret = self.__class__()
        ret.set_data(s)
        return ret

    def __getattribute__(self, attr):
        _data = object.__getattribute__(self, '_data')
        if attr in _data.dtype.names:
            return _data[attr]
        return object.__getattribute__(self, attr)


    def set_data(self, array):
        self._data = array

    def get_data(self):
        return self._data

    def fromlist(self, data):   # XXX: is it needed?
        self.set_data(np.asarray(data, dtype=self._data.dtype))


    def append(self, obj):
        self.set_data(np.concatenate((self._data, obj._data)))

    def insert(self, index, obj):
        self.set_data(np.insert(self._data, index, obj._data))

    def pop(self, index=-1):
        arr = self._data
        item = arr[index]
        self.set_data(np.delete(arr, index))
        return item

#    def remove(self, index):
#        self.set_data(np.delete(self.get_data(), index))








########## end of file ##########
