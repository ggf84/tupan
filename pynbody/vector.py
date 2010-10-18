#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Definition of class Vector"""

from __future__ import division
from math import sqrt

__all__ = ['Vector']


class Vector(object):
    """A base class for 3-dimensional vectors"""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        fmt = '[{0:s}, {1:s}, {2:s}]'
        return fmt.format(repr(self.x), repr(self.y), repr(self.z))

    # Comparison Methods

    def __lt__(self, obj):
        temp = self.x * self.x + self.y * self.y + self.z * self.z
        if isinstance(obj, Vector):
            return temp < obj.x * obj.x + obj.y * obj.y + obj.z * obj.z
        return temp < obj * obj

    def __le__(self, obj):
        temp = self.x * self.x + self.y * self.y + self.z * self.z
        if isinstance(obj, Vector):
            return temp <= obj.x * obj.x + obj.y * obj.y + obj.z * obj.z
        return temp <= obj * obj

    def __eq__(self, obj):
        if isinstance(obj, Vector):
            return self.x == obj.x and self.y == obj.y and self.z == obj.z
        return self.x * self.x + self.y * self.y + self.z * self.z == obj * obj

    def __ne__(self, obj):
        if isinstance(obj, Vector):
            return self.x != obj.x and self.y != obj.y and self.z != obj.z
        return self.x * self.x + self.y * self.y + self.z * self.z != obj * obj

    def __gt__(self, obj):
        temp = self.x * self.x + self.y * self.y + self.z * self.z
        if isinstance(obj, Vector):
            return temp > obj.x * obj.x + obj.y * obj.y + obj.z * obj.z
        return temp > obj * obj

    def __ge__(self, obj):
        temp = self.x * self.x + self.y * self.y + self.z * self.z
        if isinstance(obj, Vector):
            return temp >= obj.x * obj.x + obj.y * obj.y + obj.z * obj.z
        return temp >= obj * obj

    # Boolean Operation

    def __bool__(self):
        return self.x != 0 or self.y != 0 or self.z != 0

    # Container Methods

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]

    def __setitem__(self, index, value):
        temp = [self.x, self.y, self.z]
        temp[index] = value
        self.x, self.y, self.z = temp

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __reversed__(self):
        yield self.z
        yield self.y
        yield self.x

    def __contains__(self, obj):
        return obj in (self.x, self.y, self.z)

    # Binary Arithmetic Operations

    def __add__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x + obj.x, self.y + obj.y, self.z + obj.z)
        return Vector(self.x + obj, self.y + obj, self.z + obj)

    def __sub__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x - obj.x, self.y - obj.y, self.z - obj.z)
        return Vector(self.x - obj, self.y - obj, self.z - obj)

    def __mul__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x * obj.x, self.y * obj.y, self.z * obj.z)
        return Vector(self.x * obj, self.y * obj, self.z * obj)

    def __truediv__(self, obj):  # PY3K
        if isinstance(obj, Vector):
            return Vector(self.x / obj.x, self.y / obj.y, self.z / obj.z)
        return Vector(self.x / obj, self.y / obj, self.z / obj)

    def __floordiv__(self, obj):  # PY3K
        if isinstance(obj, Vector):
            return Vector(self.x // obj.x, self.y // obj.y, self.z // obj.z)
        return Vector(self.x // obj, self.y // obj, self.z // obj)

    def __mod__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x % obj.x, self.y % obj.y, self.z % obj.z)
        return Vector(self.x % obj, self.y % obj, self.z % obj)

    def __divmod__(self, obj):  # PY3K
        if isinstance(obj, Vector):
            return (Vector(self.x // obj.x, self.y // obj.y, self.z // obj.z),
                    Vector(self.x % obj.x, self.y % obj.y, self.z % obj.z))
        return (Vector(self.x // obj, self.y // obj, self.z // obj),
                Vector(self.x % obj, self.y % obj, self.z % obj))

    def __pow__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x ** obj.x, self.y ** obj.y, self.z ** obj.z)
        return Vector(self.x ** obj, self.y ** obj, self.z ** obj)

    def __lshift__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x << obj.x, self.y << obj.y, self.z << obj.z)
        return Vector(self.x << obj, self.y << obj, self.z << obj)

    def __rshift__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x >> obj.x, self.y >> obj.y, self.z >> obj.z)
        return Vector(self.x >> obj, self.y >> obj, self.z >> obj)

    def __and__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x & obj.x, self.y & obj.y, self.z & obj.z)
        return Vector(self.x & obj, self.y & obj, self.z & obj)

    def __xor__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x ^ obj.x, self.y ^ obj.y, self.z ^ obj.z)
        return Vector(self.x ^ obj, self.y ^ obj, self.z ^ obj)

    def __or__(self, obj):
        if isinstance(obj, Vector):
            return Vector(self.x | obj.x, self.y | obj.y, self.z | obj.z)
        return Vector(self.x | obj, self.y | obj, self.z | obj)

    # Binary Arithmetic Operations (with reflected operands)

    def __radd__(self, obj):
        return Vector(obj + self.x, obj + self.y, obj + self.z)

    def __rsub__(self, obj):
        return Vector(obj - self.x, obj - self.y, obj - self.z)

    def __rmul__(self, obj):
        return Vector(obj * self.x, obj * self.y, obj * self.z)

    def __rtruediv__(self, obj):  # PY3K
        return Vector(obj / self.x, obj / self.y, obj / self.z)

    def __rfloordiv__(self, obj):  # PY3K
        return Vector(obj // self.x, obj // self.y, obj // self.z)

    def __rmod__(self, obj):
        return Vector(obj % self.x, obj % self.y, obj % self.z)

    def __rdivmod__(self, obj):  # PY3K
        return (Vector(obj // self.x, obj // self.y, obj // self.z),
                Vector(obj % self.x, obj % self.y, obj % self.z))

    def __rpow__(self, obj):
        return Vector(obj ** self.x, obj ** self.y, obj ** self.z)

    def __rlshift__(self, obj):
        return Vector(obj << self.x, obj << self.y, obj << self.z)

    def __rrshift__(self, obj):
        return Vector(obj >> self.x, obj >> self.y, obj >> self.z)

    def __rand__(self, obj):
        return Vector(obj & self.x, obj & self.y, obj & self.z)

    def __rxor__(self, obj):
        return Vector(obj ^ self.x, obj ^ self.y, obj ^ self.z)

    def __ror__(self, obj):
        return Vector(obj | self.x, obj | self.y, obj | self.z)

    # Augmented Arithmetic Assignments

    def __iadd__(self, obj):
        if isinstance(obj, Vector):
            self.x += obj.x
            self.y += obj.y
            self.z += obj.z
        else:
            self.x += obj
            self.y += obj
            self.z += obj
        return self

    def __isub__(self, obj):
        if isinstance(obj, Vector):
            self.x -= obj.x
            self.y -= obj.y
            self.z -= obj.z
        else:
            self.x -= obj
            self.y -= obj
            self.z -= obj
        return self

    def __imul__(self, obj):
        if isinstance(obj, Vector):
            self.x *= obj.x
            self.y *= obj.y
            self.z *= obj.z
        else:
            self.x *= obj
            self.y *= obj
            self.z *= obj
        return self

    def __itruediv__(self, obj):  # PY3K
        if isinstance(obj, Vector):
            self.x /= obj.x
            self.y /= obj.y
            self.z /= obj.z
        else:
            self.x /= obj
            self.y /= obj
            self.z /= obj
        return self

    def __ifloordiv__(self, obj):  # PY3K
        if isinstance(obj, Vector):
            self.x //= obj.x
            self.y //= obj.y
            self.z //= obj.z
        else:
            self.x //= obj
            self.y //= obj
            self.z //= obj
        return self

    def __imod__(self, obj):
        if isinstance(obj, Vector):
            self.x %= obj.x
            self.y %= obj.y
            self.z %= obj.z
        else:
            self.x %= obj
            self.y %= obj
            self.z %= obj
        return self

    def __ipow__(self, obj):
        if isinstance(obj, Vector):
            self.x **= obj.x
            self.y **= obj.y
            self.z **= obj.z
        else:
            self.x **= obj
            self.y **= obj
            self.z **= obj
        return self

    def __ilshift__(self, obj):
        if isinstance(obj, Vector):
            self.x <<= obj.x
            self.y <<= obj.y
            self.z <<= obj.z
        else:
            self.x <<= obj
            self.y <<= obj
            self.z <<= obj
        return self

    def __irshift__(self, obj):
        if isinstance(obj, Vector):
            self.x >>= obj.x
            self.y >>= obj.y
            self.z >>= obj.z
        else:
            self.x >>= obj
            self.y >>= obj
            self.z >>= obj
        return self

    def __iand__(self, obj):
        if isinstance(obj, Vector):
            self.x &= obj.x
            self.y &= obj.y
            self.z &= obj.z
        else:
            self.x &= obj
            self.y &= obj
            self.z &= obj
        return self

    def __ixor__(self, obj):
        if isinstance(obj, Vector):
            self.x ^= obj.x
            self.y ^= obj.y
            self.z ^= obj.z
        else:
            self.x ^= obj
            self.y ^= obj
            self.z ^= obj
        return self

    def __ior__(self, obj):
        if isinstance(obj, Vector):
            self.x |= obj.x
            self.y |= obj.y
            self.z |= obj.z
        else:
            self.x |= obj
            self.y |= obj
            self.z |= obj
        return self

    # Unary Arithmetic Operations

    def __pos__(self):
        return Vector(+self.x, +self.y, +self.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __invert__(self):
        return Vector(~self.x, ~self.y, ~self.z)

    def __abs__(self):
        return Vector(abs(self.x), abs(self.y), abs(self.z))

    # Square and Norm Vector Operations

    def square(self):
        """Returns the square norm of the vector"""
        return (self.x * self.x + self.y * self.y + self.z * self.z)

    def norm(self):
        """Returns the norm of the vector"""
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    # Unit Vector Operations

    def unit_vector(self):
        """Returns its unit vector"""
        temp = sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        return Vector(self.x / temp, self.y / temp, self.z / temp)

    def normalize(self):
        """Returns the normalized vector"""
        temp = sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x, self.y, self.z = self.x / temp, self.y / temp, self.z / temp
        return self

    # Vector Multiplication Operations

    def dot(self, obj):
        """Returns the dot vector product"""
        return (self.x * obj.x + self.y * obj.y + self.z * obj.z)

    def cross(self, obj):
        """Returns the cross vector product"""
        return Vector(self.y * obj.z - self.z * obj.y,
                      self.z * obj.x - self.x * obj.z,
                      self.x * obj.y - self.y * obj.x)

    # Copy Vector Operation

    def copy(self):
        """Returns a copy of the vector"""
        return Vector(self.x, self.y, self.z)


########## end of file ##########
