#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np



class Pbase(object):
    """
    A common base class for all particle types
    """

    def __init__(self, dtype):
        self._data = np.zeros(10, dtype=dtype)
        self._update_fields()

    def _update_fields(self):
        for attr in self._data.dtype.names:
            item = self._data[attr]
            if len(item) == 1:
                item = item[0]
            setattr(self, attr, item)

    def __iter__(self):
#        obj = []
#        for i in xrange(len(self._data)):
#            item = self.__class__()
#            item.set_data(self[i].get_data())
#            obj.append(item)
#        return iter(obj)
        return iter(self._data)

    def __getitem__(self, index):
        if not isinstance(index, slice):
            index = slice(index, index+1)
        s = self._data[index]
        ret = self.__class__()
        ret.set_data(s)
        return ret

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self._data)

    def __reversed__(self):
#        return reversed(list(iter(self)))
        return reversed(self._data)

    def fromlist(self, data):   # XXX: is it needed?
        self.set_data(np.asarray(data, dtype=self._data.dtype))

    def set_data(self, array):
        self._data = array
        self._update_fields()

    def get_data(self):
        return self._data

    def append(self, obj):
        self.set_data(np.concatenate((self._data, obj)))

    def pop(self, index=-1):
        arr = self._data
        item = arr[index]
        self.set_data(np.delete(arr, index))
        return item

    def insert(self, index, obj):
        self.set_data(np.insert(self._data, index, obj))

#    def remove(self, index):
#        self.set_data(np.delete(self.get_data(), index))

    # gravity methods

    def set_phi(self, objs):
        raise NotImplementedError

    def set_acc(self, objs):
        raise NotImplementedError

    # energy methods

    def get_ekin(self):
        raise NotImplementedError

    def get_epot(self):
        raise NotImplementedError

    def get_energies(self):
        raise NotImplementedError

    # CoM methods

    def get_rCoM(self):
        raise NotImplementedError

    def get_vCoM(self):
        raise NotImplementedError

    def reset_CoM(self):
        raise NotImplementedError

    def shift_CoM(self, rshift, vshift):
        raise NotImplementedError

    # momentum methods

    def get_linear_mom(self):
        raise NotImplementedError

    def get_angular_mom(self):
        raise NotImplementedError

    def get_angular_mom_squared(self):
        amom = self.get_angular_mom()
        return np.dot(amom, amom)


























#class Pbase(object):
#    """
#    A common base class for all particle types
#    """

#    def __init__(self, dtype):
#        self._dtype = dtype
#        for (k, v) in dict(self._dtype).iteritems():
#            setattr(self, k, np.zeros(10, dtype=v))

#    def __iter__(self):
#        return iter(self.get_data())

#    def __len__(self):
#        return len(self.get_data())

#    def __repr__(self):
#        return '{0}'.format(self.get_data())

#    def __reversed__(self):
#        return reversed(self.get_data())

#    def __getitem__(self, index):
#        s = self.get_data()[index]
#        if not isinstance(s, np.ndarray):
#            s = np.asarray([s], dtype=self._dtype)
#        ret = self.__class__()
#        ret.set_data(s)
#        return ret

#    def set_data(self, array):
#        for attr in dict(self._dtype).iterkeys():
#            setattr(self, attr, array[attr])

#    def get_data(self):
#        array = np.empty(len(self.index), dtype=self._dtype)
#        for attr in dict(self._dtype).iterkeys():
#            array[attr] = getattr(self, attr)
#        return array

#    def append(self, obj):
#        self.set_data(np.concatenate((self.get_data(), obj)))

#    def pop(self, index=-1):
#        arr = self.get_data()
#        item = arr[index]
#        self.set_data(np.delete(arr, index))
#        return item

##    def remove(self, index):
##        self.set_data(np.delete(self.get_data(), index))

##    def concatenate(self, obj):
##        ret = self.__class__()
##        ret.set_data(np.concatenate((self.get_data(), obj)))
##        return ret

#    # gravity methods

#    def set_phi(self, objs):
#        raise NotImplementedError

#    def set_acc(self, objs):
#        raise NotImplementedError

#    # energy methods

#    def get_ekin(self):
#        raise NotImplementedError

#    def get_epot(self):
#        raise NotImplementedError

#    def get_energies(self):
#        ekin = self.get_ekin()
#        epot = self.get_epot()
#        etot = ekin + epot
#        return (ekin, epot, etot)

#    # CoM methods

#    def get_rCoM(self):
#        raise NotImplementedError

#    def get_vCoM(self):
#        raise NotImplementedError

#    def reset_CoM(self):
#        raise NotImplementedError

#    def shift_CoM(self, rshift, vshift):
#        raise NotImplementedError

#    # momentum methods

#    def get_linear_mom(self):
#        raise NotImplementedError

#    def get_angular_mom(self):
#        raise NotImplementedError

#    def get_angular_mom_squared(self):
#        amom = self.get_angular_mom()
#        return np.dot(amom, amom)




########## end of file ##########
