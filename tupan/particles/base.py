# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import abc
import copy
import numpy as np
from ..config import Ctype


class MetaParticles(abc.ABCMeta):
    def __init__(cls, *args, **kwargs):
        super(MetaParticles, cls).__init__(*args, **kwargs)

        attrs = {}
        attrs.update(**cls.default_attrs)
        attrs.update(**cls.extra_attrs)
        setattr(cls, 'attrs', attrs)
        setattr(cls, 'name', cls.__name__.lower())


class Particles(metaclass=MetaParticles):
    """

    """
    part_type = 0
    default_attrs = {
        'pid': ('{nb}', 'uint_t', 'particle id'),
        'mass': ('{nb}', 'real_t', 'mass'),
        'eps2': ('{nb}', 'real_t', 'squared smoothing parameter'),
        'rdot': ('10, {nd}, {nb}', 'real_t', 'position and its derivatives'),
        'time': ('{nb}', 'real_t', 'current time'),
        'nstep': ('{nb}', 'uint_t', 'step number'),
    }

    extra_attrs = {
        'phi': ('{nb}', 'real_t', 'gravitational potential'),
        'fdot': ('4, {nd}, {nb}', 'real_t', 'force and its derivatives'),
        'tstep': ('{nb}', 'real_t', 'time step'),
        'tstep_sum': ('{nb}', 'real_t', 'auxiliary time step'),
    }

    def __init__(self, n=0):
        self.n = int(n)

    @classmethod
    def from_attrs(cls, **attrs):
        obj = cls.__new__(cls)
        vars(obj).update(**attrs)
        obj.n = len(obj.pid)
        return obj

    def register_attribute(self, attr, shape, sctype, doc=''):
        if attr not in self.attrs:
            self.attrs.update(**{attr: (shape, sctype, doc)})

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return repr(vars(self))

    def __str__(self):
        fmt = self.name + '(['
        if self.n:
            for attr in self.attrs:
                value = getattr(self, attr)
                fmt += '\n\t{0}: {1},'.format(attr, value)
            fmt += '\n'
        fmt += '])'
        return fmt

    def __contains__(self, idx):
        return idx in self.pid

    def __len__(self):
        return self.n

    def __add__(self, other):
        if other.n:
            data = {}
            concatenate = np.concatenate
            for attr in self.attrs:
                arrays = [getattr(self, attr), getattr(other, attr)]
                data[attr] = concatenate(arrays, -1)  # along last dimension
            return self.from_attrs(**data)
        return self

    __radd__ = __add__

    def split_by(self, mask_function):
        mask = mask_function(self)
        if all(mask):
            return self, type(self)()
        if not any(mask):
            return type(self)(), self

        a, b = {}, {}
        for attr in self.attrs:
            array = getattr(self, attr)
            a[attr] = np.array(array[...,  mask], copy=False, order='C')
            b[attr] = np.array(array[..., ~mask], copy=False, order='C')
        return (self.from_attrs(**a),
                self.from_attrs(**b))

    def __getitem__(self, index):
        index = ((Ellipsis, index, None)
                 if isinstance(index, int)
                 else (Ellipsis, index))
        data = {}
        for attr in self.attrs:
            value = getattr(self, attr)[index]
            value = np.array(value, copy=False, order='C')
            data[attr] = value
        return self.from_attrs(**data)

    def __setitem__(self, index, data):
        for attr in self.attrs:
            array = getattr(self, attr)
            array[..., index] = getattr(data, attr)

    def __getattr__(self, attr):
        if attr not in self.attrs:
            raise AttributeError(attr)
        shape, sctype, _ = self.attrs[attr]
        shape = eval(shape.format(nd=3, nb=self.n))
        dtype = vars(Ctype)[sctype]
        value = np.zeros(shape, dtype=dtype)
        setattr(self, attr, value)
        return value

    def astype(self, cls):
        obj = cls(self.n)
        obj.set_state(self.get_state())
        return obj

    def get_state(self):
        data = {}
        for attr in self.attrs:
            value = getattr(self, attr)
            data[attr] = value
        return data

    def set_state(self, data):
        for attr in self.attrs:
            if attr in data:
                value = getattr(self, attr)
                value[...] = data[attr]


class Bodies(Particles):
    """

    """
    pass


class Stars(Particles):
    """

    """
    part_type = 1
    default_attrs = Particles.default_attrs.copy()
    extra_attrs = Particles.extra_attrs.copy()

    default_attrs.update(**{
        'age': ('{nb}', 'real_t', 'age'),
        'spin': ('{nd}, {nb}', 'real_t', 'spin'),
        'radius': ('{nb}', 'real_t', 'radius'),
        'metallicity': ('{nb}', 'real_t', 'metallicity'),
    })


class Planets(Particles):
    """

    """
    part_type = 2
    default_attrs = Particles.default_attrs.copy()
    extra_attrs = Particles.extra_attrs.copy()

    default_attrs.update(**{
        'spin': ('{nd}, {nb}', 'real_t', 'spin'),
        'radius': ('{nb}', 'real_t', 'radius'),
    })


class Blackholes(Particles):
    """

    """
    part_type = 3
    default_attrs = Particles.default_attrs.copy()
    extra_attrs = Particles.extra_attrs.copy()

    default_attrs.update(**{
        'spin': ('{nd}, {nb}', 'real_t', 'spin'),
        'radius': ('{nb}', 'real_t', 'radius'),
    })


class Gas(Particles):
    """

    """
    part_type = 4
    default_attrs = Particles.default_attrs.copy()
    extra_attrs = Particles.extra_attrs.copy()

    default_attrs.update(**{
        'density': ('{nb}', 'real_t', 'density at particle position'),
        'pressure': ('{nb}', 'real_t', 'pressure at particle position'),
        'viscosity': ('{nb}', 'real_t', 'viscosity at particle position'),
        'temperature': ('{nb}', 'real_t', 'temperature at particle position'),
    })


# -- End of File --
