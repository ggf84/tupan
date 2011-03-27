#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import random
import math

from pynbody.lib.decorators import selftimer
from pynbody.particles import (Body, Particles)
from pynbody.io import HDF5IO


def scale_mass(bodies, m_scale):
    bodies.mass *= m_scale

def scale_pos(bodies, r_scale):
    bodies.pos *= r_scale

def scale_vel(bodies, v_scale):
    bodies.vel *= v_scale

def scale_to_virial(bodies, ekin, epot, etot):
    scale_vel(bodies, math.sqrt(-0.5*epot/ekin))
    ekin = -0.5*epot
    etot = ekin + epot
    scale_pos(bodies, etot/(-0.25))
    scale_vel(bodies, math.sqrt(-0.25/etot))

def scale_to_nbody_units(bodies):
    scale_mass(bodies, 1.0/bodies.get_mass())

    bodies.set_phi(bodies)
    ekin = bodies.get_ekin()
    epot = bodies.get_epot()
    etot = ekin + epot
#    print(ekin, epot, etot)

    scale_to_virial(bodies, ekin, epot, etot)

    bodies.set_phi(bodies)
    ekin = bodies.get_ekin()
    epot = bodies.get_epot()
    etot = ekin + epot
#    print(ekin, epot, etot)







class Plummer(object):
    """  """

    def __init__(self, num, mfrac=0.999, seed=None):
        self.num = num
        self.mfrac = mfrac
        self.bodies = Body(num)
        random.seed(seed)

    def set_mass(self):
        return 1.0 / self.num   # equal masses

    def set_pos(self, irand):
        m_min = (irand * self.mfrac) / self.num
        m_max = ((irand+1) * self.mfrac) / self.num
        mrand = random.uniform(m_min, m_max)
        radius = 1.0 / math.sqrt(math.pow(mrand, -2.0/3.0) - 1.0)
        theta = math.acos(random.uniform(-1.0, 1.0))
        phi = random.uniform(0.0, 2.0*math.pi)
        rx = radius * math.sin(theta) * math.cos(phi)
        ry = radius * math.sin(theta) * math.sin(phi)
        rz = radius * math.cos(theta)
        return [rx, ry, rz]

    def set_vel(self, pot):                              # E = T+U
        q = 1.0                                          # 2*T+U = 0
        g = 2.0                                          # E = U/2 = -T
        while (g > 2 * (1 - q)):                         # v^2 = 2*(e-u)
            q = random.uniform(-0.5, 1.0)                #     ~ g*u
            g = random.uniform(0.0, 3.0)                 # Emin < E < Emax
        velocity = math.sqrt(-g * pot)                   # Emin = U
        theta = math.acos(random.uniform(-1.0, 1.0))     # Emax = T
        phi = random.uniform(0.0, 2.0*math.pi)           # de = [-1/2, 1]*u
        vx = velocity * math.sin(theta) * math.cos(phi)  #    ~ q*u
        vy = velocity * math.sin(theta) * math.sin(phi)  # v^2 = 2*(1-q)*u
        vz = velocity * math.cos(theta)                  # g = 2*(1-q)
        return [vx, vy, vz]                              # g = [0, 3]

    @selftimer
    def set_bodies(self):
        """  """
        n = self.num
        ilist = list(range(n))

        # set index
        self.bodies.index[:] = ilist
        # set mass
        self.bodies.mass[:] = [self.set_mass() for i in range(n)]
        # set pos
        self.bodies.pos[:] = [self.set_pos(i) for i in random.sample(ilist,n)]
        # set phi
        self.bodies.set_phi(self.bodies)
        # set vel
        self.bodies.vel[:] = [self.set_vel(p) for p in self.bodies.phi]


    def make_plummer(self):
        self.set_bodies()
        self.bodies.reset_center_of_mass()
        scale_to_nbody_units(self.bodies)


    def write_snapshot(self, fname='plummer.hdf5'):
        data = Particles()
        data.set_members(self.bodies)
        io = HDF5IO(fname, 'w')
        io.write_snapshot(data)


########## end of file ##########
