#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np


__all__ = ['Pbase']


class Pbase(object):
    """
    A base class implementing common functionalities for all types of particles.
    """
    def __init__(self, numobjs, dtype):
        self._dtype = dtype
        self._data = None
        if numobjs >= 0:
            self._data = np.zeros(numobjs, dtype)
            fields = {}
            for attr in self._dtype['names']:
                fields[attr] = self._data[attr]
            self.__dict__.update(fields)


    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self._data)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __reversed__(self):
        return reversed(self._data)


    def __getitem__(self, index):
#        if not isinstance(index, slice):
#            index = slice(index, index+1)
        s = self._data[index].copy()
        ret = self.__class__()
        ret.__dict__.update(self.__dict__)
        ret.set_data(s)
        return ret

#    def __getattribute__(self, attr):
#        _dtype = object.__getattribute__(self, '_dtype')
#        if attr in _dtype['names']:
#            _data = object.__getattribute__(self, '_data')
#            return _data[attr]
#        return object.__getattribute__(self, attr)


    def set_data(self, array):
        self._data = array
        fields = {}
        for attr in self._dtype['names']:
            fields[attr] = self._data[attr]
        self.__dict__.update(fields)


    def get_data(self):
        return self._data

    def copy(self):
        ret = self.__class__()
        ret.__dict__.update(self.__dict__)
        ret.set_data(self._data.copy())
        return ret

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


    def fromlist(self, data, dtype):
        self._dtype = dtype
        self._data = None
        if len(data) > 0:
            self._data = np.array(data, dtype)
            fields = {}
            for attr in self._dtype['names']:
                fields[attr] = self._data[attr]
            self.__dict__.update(fields)






########## end of file ##########
