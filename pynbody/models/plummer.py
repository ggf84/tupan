#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import (print_function, with_statement)
import math
import random

from ..universe import Body
from ..vector import Vector

random.seed(1)


class Plummer(object):
    """  """

    def __init__(self, num):
        self.num = num
        self.mfrac = 0.999
        self.rscale = (3.0 * math.pi / 16.0)
        self.vscale = 1.0 / math.sqrt(self.rscale)

    def genmass(self):
        pass

    def genpos(self):
        particle = []
        index = list(range(self.num))
        random.shuffle(index)
        for i in sorted(index):
            # position coordinates
            m_min = (i * self.mfrac) / self.num
            m_max = ((i+1) * self.mfrac) / self.num
            mrand = random.uniform(m_min, m_max)
            radius = 1.0 / math.sqrt(math.pow(mrand, -2.0/3.0) - 1.0)
            theta = math.acos(random.uniform(-1.0, 1.0))
            phi = random.uniform(0.0, 2.0*math.pi)
            rx = radius * math.sin(theta) * math.cos(phi)
            ry = radius * math.sin(theta) * math.sin(phi)
            rz = radius * math.cos(theta)

            # velocity coordinates
            q = 0.0
            g = 0.092211
            while (g > (q * q * math.pow(1.0 - q * q, 3.5))):
                q = random.uniform(0.0, 1.0)
                g = random.uniform(0.0, 0.092211)
            velocity = q * math.sqrt(2.0) * math.pow(1.0 + radius**2, -0.25)
            theta = math.acos(random.uniform(-1.0, 1.0))
            phi = random.uniform(0.0, 2.0*math.pi)
            vx = velocity * math.sin(theta) * math.cos(phi)
            vy = velocity * math.sin(theta) * math.sin(phi)
            vz = velocity * math.cos(theta)

            # rescaling to N-body units
            pos = Vector(rx, ry, rz)
            pos *= self.rscale
            vel = Vector(vx, vy, vz)
            vel *= self.vscale

            mass = 1.0/self.num

            data = (index[i], 0, 0.0, 0.0, mass, pos, vel)

            particle.append(Body(data))
        return sorted(particle)

    def genvel(self, body):
        for b in body:                                       # E = T+U
            q = 1.0                                          # 2*T+U = 0
            g = 2.0                                          # E = U/2 = -T
            while (g > 2 * (1 - q)):                         # v^2 = 2*(e-u)
                q = random.uniform(-0.5, 1.0)                #     ~ g*u
                g = random.uniform(0.0, 3.0)                 # Emin < E < Emax
            velocity = math.sqrt(-g * b.get_pot(body))       # Emin = U
            theta = math.acos(random.uniform(-1.0, 1.0))     # Emax = T
            phi = random.uniform(0.0, 2.0*math.pi)           # de = [-1/2, 1]*u
            vx = velocity * math.sin(theta) * math.cos(phi)  #    ~ q*u
            vy = velocity * math.sin(theta) * math.sin(phi)  # v^2 = 2*(1-q)*u
            vz = velocity * math.cos(theta)                  # g = 2*(1-q)
            b.vel = Vector(vx, vy, vz)                       # g = [0, 3]


    def make_plummer(self):
        pass

    def dump_to_txt(self):
        with open('data.txt', 'w') as f:
            particle = self.genpos()
            self.genvel(particle)
            for b in particle:
                pot = b.get_pot(particle)
                print(b.index, b.pos.x, b.pos.y, b.pos.z,
                      b.vel.x, b.vel.y, b.vel.z, pot, file=f)





########## end of file ##########
