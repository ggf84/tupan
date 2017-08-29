# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import abc
import copy
import numpy as np
from itertools import count
from ..lib.utils.ctype import Ctype


part_type = count()


class Attributes(object):
    """This class holds common attribute descriptions for particles.

    The descriptions are composed by: 'name', 'shape', 'ctype' and 'doc'.

    """
    default = [
        ('pid', '{nb}', 'uint_t', 'particle id'),
        ('mass', '{nb}', 'real_t', 'mass'),
        ('eps2', '{nb}', 'real_t', 'squared smoothing parameter'),
        ('rdot', '10, {nd}, {nb}', 'real_t', 'position and its derivatives'),
        ('time', '{nb}', 'real_t', 'current time'),
        ('nstep', '{nb}', 'uint_t', 'step number'),
    ]

    extra = [
        ('phi', '{nb}', 'real_t', 'gravitational potential'),
        ('fdot', '4, {nd}, {nb}', 'real_t', 'force and its derivatives'),
        ('tstep', '{nb}', 'real_t', 'time step'),
        ('tstep_sum', '{nb}', 'real_t', 'auxiliary time step'),
    ]


class MetaParticle(abc.ABCMeta):
    def __init__(cls, *args, **kwargs):
        super(MetaParticle, cls).__init__(*args, **kwargs)

        if hasattr(cls, 'part_type'):
            setattr(cls, 'part_type', next(part_type))

        if hasattr(cls, 'name'):
            setattr(cls, 'name', cls.__name__.lower())

            attr_descrs = []
            if hasattr(cls, 'default_attr_descr'):
                attr_descrs += cls.default_attr_descr

            default_attr_names = {name: (shape, sctype, doc)
                                  for name, shape, sctype, doc in attr_descrs}
            setattr(cls, 'default_attr_names', default_attr_names)

            if hasattr(cls, 'extra_attr_descr'):
                attr_descrs += cls.extra_attr_descr

            attr_names = {name: (shape, sctype, doc)
                          for name, shape, sctype, doc in attr_descrs}
            setattr(cls, 'attr_names', attr_names)


class AbstractParticle(metaclass=MetaParticle):
    """

    """
    def __init__(self, n=0):
        self.n = int(n)

    @classmethod
    def from_attrs(cls, **attrs):
        obj = cls.__new__(cls)
        vars(obj).update(**attrs)
        obj.n = len(obj.pid)
        return obj

    def register_attribute(self, name, shape, sctype, doc=''):
        if name not in self.attr_names:
            self.attr_names[name] = (shape, sctype, doc)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return repr(vars(self))

    def __str__(self):
        fmt = self.name + '(['
        if self.n:
            for name in self.attr_names:
                value = getattr(self, name)
                fmt += '\n\t{0}: {1},'.format(name, value)
            fmt += '\n'
        fmt += '])'
        return fmt

    def __contains__(self, idx):
        return idx in self.pid

    def __len__(self):
        return self.n

    def __add__(self, other):
        if other.n:
            attrs = {}
            concatenate = np.concatenate
            for name in self.attr_names:
                arrays = [getattr(self, name), getattr(other, name)]
                attrs[name] = concatenate(arrays, -1)  # along last dimension
            return self.from_attrs(**attrs)
        return self

    __radd__ = __add__

    def split_by(self, mask_function):
        mask = mask_function(self)
        if all(mask):
            return self, type(self)()
        if not any(mask):
            return type(self)(), self

        a, b = {}, {}
        for name in self.attr_names:
            array = getattr(self, name)
            a[name] = np.array(array[...,  mask], copy=False, order='C')
            b[name] = np.array(array[..., ~mask], copy=False, order='C')
        return (self.from_attrs(**a),
                self.from_attrs(**b))

    def __getitem__(self, index):
        index = ((Ellipsis, index, None)
                 if isinstance(index, int)
                 else (Ellipsis, index))
        attrs = {}
        for name in self.attr_names:
            value = getattr(self, name)[index]
            value = np.array(value, copy=False, order='C')
            attrs[name] = value
        return self.from_attrs(**attrs)

    def __setitem__(self, index, value):
        for name in self.attr_names:
            attr = getattr(self, name)
            attr[..., index] = getattr(value, name)

    def __getattr__(self, name):
        if name not in self.attr_names:
            raise AttributeError(name)
        shape, sctype, _ = self.attr_names[name]
        shape = eval(shape.format(nd=3, nb=self.n))
        dtype = vars(Ctype)[sctype]
        value = np.zeros(shape, dtype=dtype)
        setattr(self, name, value)
        return value

    def astype(self, cls):
        obj = cls(self.n)
        obj.set_state(self.get_state())
        return obj

    def get_state(self):
        data = {}
        for name in self.attr_names:
            value = getattr(self, name)
            data[name] = value
        return data

    def set_state(self, data):
        for name in self.attr_names:
            if name in data:
                value = getattr(self, name)
                value[...] = data[name]


# -- End of File --
