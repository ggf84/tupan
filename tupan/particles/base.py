# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import abc
import copy
import numpy as np
from ..config import Ctype
from ..lib.extensions import ArrayWrapper


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
        'pos': ('{nd}, {nb}', 'real_t', 'position'),
        'vel': ('{nd}, {nb}', 'real_t', 'velocity'),
        'time': ('{nb}', 'real_t', 'current time'),
        'nstep': ('{nb}', 'uint_t', 'step number'),
    }

    extra_attrs = {
        'phi': ('{nb}', 'real_t', 'gravitational potential'),
        'acc': ('{nd}, {nb}', 'real_t', 'acceleration'),
        'jrk': ('{nd}, {nb}', 'real_t', '1st derivative of acc'),
        'snp': ('{nd}, {nb}', 'real_t', '2nd derivative of acc'),
        'crk': ('{nd}, {nb}', 'real_t', '3rd derivative of acc'),
        'ad4': ('{nd}, {nb}', 'real_t', '4th derivative of acc'),
        'ad5': ('{nd}, {nb}', 'real_t', '5th derivative of acc'),
        'ad6': ('{nd}, {nb}', 'real_t', '6th derivative of acc'),
        'ad7': ('{nd}, {nb}', 'real_t', '7th derivative of acc'),
        'f0': ('{nd}, {nb}', 'real_t', 'auxiliary acc'),
        'f1': ('{nd}, {nb}', 'real_t', 'auxiliary jrk'),
        'f2': ('{nd}, {nb}', 'real_t', 'auxiliary snp'),
        'f3': ('{nd}, {nb}', 'real_t', 'auxiliary crk'),
        'tstep': ('{nb}', 'real_t', 'time step'),
        'tstep_sum': ('{nb}', 'real_t', 'auxiliary time step'),
    }

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.n = 0
        obj.data = {}
        return obj

    def __init__(self, n=0):
        self.n = int(n)
        self.data = {}
        for attr, (shape, sctype, _) in self.attrs.items():
            shape = eval(shape.format(nd=3, nb=self.n))
            dtype = vars(Ctype)[sctype]
            array = np.zeros(shape, dtype=dtype)
            self.data[attr] = ArrayWrapper(array)

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    @classmethod
    def from_attrs(cls, **attrs):
        obj = cls.__new__(cls)
        obj.data.update(**attrs)
        obj.n = len(obj.pid)
        return obj

    def register_attribute(self, attr, shape, sctype, doc=''):
        if attr not in self.attrs:
            self.attrs[attr] = (shape, sctype, doc)
            shape = eval(shape.format(nd=3, nb=self.n))
            dtype = vars(Ctype)[sctype]
            array = np.zeros(shape, dtype=dtype)
            self.data[attr] = ArrayWrapper(array)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return repr(vars(self))

    def __str__(self):
        fmt = self.name + '(['
        if self.n:
            for k, v in self.data.items():
                fmt += '\n\t{0}: {1},'.format(k, v)
            fmt += '\n'
        fmt += '])'
        return fmt

    def __contains__(self, idx):
        return idx in self.pid

    def __len__(self):
        return self.n

    def __add__(self, other):
        if not other.n:
            return self
        if not self.n:
            return other
        data = {}
        for k in set([*self.data, *other.data]):
            if k not in other.data:
                data[k] = self.data[k]
            elif k not in self.data:
                data[k] = other.data[k]
            else:
                u, v = self.data[k].ary, other.data[k].ary
                ary = np.concatenate([u, v], axis=-1)
                data[k] = ArrayWrapper(ary)
        return self.from_attrs(**data)

    __radd__ = __add__

    def __getitem__(self, index):
        try:
            data = {}
            for k, v in self.data.items():
                ary = v.ary.compress(index, axis=-1)
                data[k] = ArrayWrapper(ary)
            return self.from_attrs(**data)
        except ValueError:
            index = ((Ellipsis, index, None)
                     if isinstance(index, int)
                     else (Ellipsis, index))
            data = {}
            for k, v in self.data.items():
                ary = np.array(v.ary[index], copy=False, order='C')
                data[k] = ArrayWrapper(ary)
            return self.from_attrs(**data)

    def __setitem__(self, index, obj):
        index = ((Ellipsis, index, None)
                 if isinstance(index, int)
                 else (Ellipsis, index))
        for k, v in self.data.items():
            v.ary[index] = obj.data[k].ary

    def __getattr__(self, attr):
        try:
            return self.data[attr].ary
        except KeyError:
            raise AttributeError(attr)

    def astype(self, cls):
        obj = cls(self.n)
        obj.set_state(self.get_state())
        return obj

    def get_state(self):
        data = {}
        for k, v in self.data.items():
            data[k] = v
        return data

    def set_state(self, data):
        for k, v in self.data.items():
            if k in data:
                v.ary[...] = data[k].ary


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
