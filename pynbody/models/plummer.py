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

    def __init__(self, num, mfrac=0.999):
        self.num = num
        self.mfrac = mfrac

    def scale_mass(self, bodies, m_scale):
        for b in bodies:
            b.mass *= m_scale

    def scale_pos(self, bodies, r_scale):
        for b in bodies:
            b.pos *= r_scale

    def scale_vel(self, bodies, v_scale):
        for b in bodies:
            b.vel *= v_scale

    def scale_to_virial(self, bodies, ek_tot, ep_tot, et_tot):
        self.scale_vel(bodies, math.sqrt(-0.5*ep_tot/ek_tot))
        ek_tot = -0.5*ep_tot
        et_tot = ek_tot + ep_tot
        self.scale_pos(bodies, et_tot/(-0.25))
        self.scale_vel(bodies, math.sqrt(-0.25/et_tot))

    def scale_to_nbody_units(self, bodies):     # XXX: modificar apos ter implementado um metodo para calcular a energia total do sistema.
        mtot = 0.0
        for b in bodies:
            mtot += b.mass
        self.scale_mass(bodies, 1.0/mtot)
        ek_tot = 0.0
        ep_tot = 0.0
        et_tot = 0.0
        for b in bodies:
            ek_tot += b.get_ekin()
            b.set_pot(bodies)
            ep_tot += b.get_epot()
            et_tot += b.get_etot()
        print(ek_tot, ep_tot, et_tot)

        self.scale_to_virial(bodies, ek_tot, ep_tot, et_tot)

        ek_tot = 0.0
        ep_tot = 0.0
        et_tot = 0.0
        for b in bodies:
            ek_tot += b.get_ekin()
            b.set_pot(bodies)
            ep_tot += b.get_epot()
            et_tot += b.get_etot()
        print(ek_tot, ep_tot, et_tot)

    def set_index(self):
        index = list(range(self.num))
        random.shuffle(index)
        return index

    def set_mass(self):
        return [1.0 for i in range(self.num)]   # equal masses

    def set_pos(self):
        pos = []
        for i in range(self.num):
            m_min = (i * self.mfrac) / self.num
            m_max = ((i+1) * self.mfrac) / self.num
            mrand = random.uniform(m_min, m_max)
            radius = 1.0 / math.sqrt(math.pow(mrand, -2.0/3.0) - 1.0)
            theta = math.acos(random.uniform(-1.0, 1.0))
            phi = random.uniform(0.0, 2.0*math.pi)
            rx = radius * math.sin(theta) * math.cos(phi)
            ry = radius * math.sin(theta) * math.sin(phi)
            rz = radius * math.cos(theta)
            pos.append(Vector(rx, ry, rz))
        return pos

    def set_vel(self, pot):
        vel = []
        for p in pot:                                        # E = T+U
            q = 1.0                                          # 2*T+U = 0
            g = 2.0                                          # E = U/2 = -T
            while (g > 2 * (1 - q)):                         # v^2 = 2*(e-u)
                q = random.uniform(-0.5, 1.0)                #     ~ g*u
                g = random.uniform(0.0, 3.0)                 # Emin < E < Emax
            velocity = math.sqrt(-g * p)                     # Emin = U
            theta = math.acos(random.uniform(-1.0, 1.0))     # Emax = T
            phi = random.uniform(0.0, 2.0*math.pi)           # de = [-1/2, 1]*u
            vx = velocity * math.sin(theta) * math.cos(phi)  #    ~ q*u
            vy = velocity * math.sin(theta) * math.sin(phi)  # v^2 = 2*(1-q)*u
            vz = velocity * math.cos(theta)                  # g = 2*(1-q)
            vel.append(Vector(vx, vy, vz))                   # g = [0, 3]
        return vel

    def set_pot(self, index, mass, pos):
        pot = []
        for (i, pos_i) in zip(index, pos):
            sumpot = 0.0
            for (j, mass_j, pos_j) in zip(index, mass, pos):
                if i != j:
                    dpos = pos_i - pos_j
                    sumpot -= mass_j / dpos.norm()
            pot.append(sumpot)
        return pot

    def make_plummer(self):
        index = self.set_index()
        mass = self.set_mass()
        pos = self.set_pos()
        pot = self.set_pot(index, mass, pos)
        vel = self.set_vel(pot)

        bodies = []
        nstep = [0 for i in range(self.num)]
        tstep = [0.0 for i in range(self.num)]
        time = [0.0 for i in range(self.num)]
        for b in zip(index, nstep, tstep, time, mass, pot, pos, vel):
            bodies.append(Body(b))

        return sorted(bodies)

    def dump_to_txt(self):
        bodies = self.make_plummer()
        # TODO: correct to CoM
        self.scale_to_nbody_units(bodies)

        with open('data.txt', 'w') as f:
            for b in bodies:
                print(b.index, b.pos.x, b.pos.y, b.pos.z,
                      b.vel.x, b.vel.y, b.vel.z, b.pot, file=f)


########## end of file ##########
