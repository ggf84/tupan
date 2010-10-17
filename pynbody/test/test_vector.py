#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from ..vector import Vector

def main():
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)
    v3 = Vector(7, 8, 9)

    print('v1 = {0}'.format(v1))
    print('v2 = {0}'.format(v2))
    print '*' * 20
    print('+v1 = {0}'.format(+v1))
    print('-v1 = {0}'.format(-v1))
    print '*' * 20
    print('v1+v2 = {0}'.format(v1 + v2))
    print('v2+v1 = {0}'.format(v2 + v1))
    print('v1+3 = {0}'.format(v1 + 3))
    print('3+v1 = {0}'.format(3 + v1))
    print('v1+3.0 = {0}'.format(v1 + 3.0))
    print('3.0+v1 = {0}'.format(3.0 + v1))
    print '*' * 20
    print('v1-v2 = {0}'.format(v1 - v2))
    print('v2-v1 = {0}'.format(v2 - v1))
    print('v1-3 = {0}'.format(v1 - 3))
    print('3-v1 = {0}'.format(3 - v1))
    print('v1-3.0 = {0}'.format(v1 - 3.0))
    print('3.0-v1 = {0}'.format(3.0 - v1))
    print '*' * 20
    print('v1*v2 = {0}'.format(v1 * v2))
    print('v2*v1 = {0}'.format(v2 * v1))
    print('v1*3 = {0}'.format(v1 * 3))
    print('3*v1 = {0}'.format(3 * v1))
    print('v1*3.0 = {0}'.format(v1 * 3.0))
    print('3.0*v1 = {0}'.format(3.0 * v1))
    print '*' * 20
    print('v1/v2 = {0}'.format(v1 / v2))
    print('v2/v1 = {0}'.format(v2 / v1))
    print('v1/3 = {0}'.format(v1 / 3))
    print('3/v1 = {0}'.format(3 / v1))
    print('v1/3.0 = {0}'.format(v1 / 3.0))
    print('3.0/v1 = {0}'.format(3.0 / v1))
    print '*' * 20
    print('v1//v2 = {0}'.format(v1 // v2))
    print('v2//v1 = {0}'.format(v2 // v1))
    print('v1//3 = {0}'.format(v1 // 3))
    print('3//v1 = {0}'.format(3 // v1))
    print('v1//3.0 = {0}'.format(v1 // 3.0))
    print('3.0//v1 = {0}'.format(3.0 // v1))
    print '*' * 20
    print('v1.dot(v2) = {0}'.format(v1.dot(v2)))
    print('v1.cross(v2) = {0}'.format(v1.cross(v2)))
    print('v1.square() = {0}'.format(v1.square()))
    print('v1.norm() = {0}'.format(v1.norm()))
    print('v1.unit_vector() = {0}'.format(v1.unit_vector()))
    print('v1.normalize() = {0}'.format(v1.copy().normalize()))
    print '*' * 40
    print '*' * 40
    print('v1 = {0}'.format(v1))
    print('v2 = {0}'.format(v2))
    print('v3 = {0}'.format(v3))
    print '*' * 20
    v3 += v1
    print('v3 += v1: {0}'.format(v3))
    v3 -= v2 / 2
    print('v3 -= v2/2: {0}'.format(v3))
    v3 *= v3
    print('v3 *= v3: {0}'.format(v3))
    v3 /= v2
    print('v3 /= v1: {0}'.format(v3))
    print '*' * 20
    v3 += 3
    print('v3 += 3: {0}'.format(v3))
    v3 -= 3
    print('v3 -= 3: {0}'.format(v3))
    v3 *= 3
    print('v3 *= 3: {0}'.format(v3))
    v3 /= 3
    print('v3 /= 3: {0}'.format(v3))
    print '*' * 20
    v3 += 3.0
    print('v3 += 3.0: {0}'.format(v3))
    v3 -= 3.0
    print('v3 -= 3.0: {0}'.format(v3))
    v3 *= 3.0
    print('v3 *= 3.0: {0}'.format(v3))
    v3 /= 3.0
    print('v3 /= 3.0: {0}'.format(v3))
    print '*' * 40

    v4 = Vector()
    v4 += (v1 + 4 * v2 / v3 - v1.unit_vector()) * 2

    s = """v4 = Vector()
v4 += (v1+4*v2/v3-v1.unit_vector())*2
v4 = {0}
"""
    print(s.format(v4))

    a, b, c = v4

    print 'a,b,c = ', a, b, c
    print v1, v2, v3, v4

    print v2, s.format(v4), v1, v3

    for i, elem in enumerate(reversed(v4)):
        print i, elem





if __name__ == "__main__":
    sys.exit(main())



########## end of file ##########
