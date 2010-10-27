#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This module holds base classes for particle types"""

from vector import Vector


class Body(object):
    """A base class for particles"""

    def __init__(self,
                 index=None,
                 nstep=None,
                 time=None,
                 tstep=None,
                 mass=None,
                 pos=None,
                 vel=None):

        if index is None:
            self.index = 0
        else:
            self.index = index

        if nstep is None:
            self.nstep = 0
        else:
            self.nstep = nstep

        if time is None:
            self.time = 0.0
        else:
            self.time = time

        if tstep is None:
            self.tstep = 0.0
        else:
            self.tstep = tstep

        if mass is None:
            self.mass = 0.0
        else:
            self.mass = mass

        if pos is None:
            self.pos = Vector(0, 0, 0)
        else:
            self.pos = pos

        if vel is None:
            self.vel = Vector(0, 0, 0)
        else:
            self.vel = vel

        self.ekin = None
        self.epot = None
        self.etot = None

    def __repr__(self):
        fmt = '[{0:s}, {1:s}, {2:s}, {3:s}, {4:s}, {5:s}, {6:s}]'
        return fmt.format(repr(self.index), repr(self.nstep), repr(self.time),
                          repr(self.tstep), repr(self.mass), repr(self.pos),
                          repr(self.vel))

    def get_acc(self, bodies):
        """get the particle's acceleration due to other bodies"""
        acc = Vector(0, 0, 0)
        for body in bodies:
            dpos = self.pos - body.pos
            dposinv3 = dpos.square() ** (-1.5)
            acc -= body.mass * dpos * dposinv3
        return acc

    def get_pot(self, bodies):
        """get the gravitational potential in the particle's location"""
        pot = 0.0
        for body in bodies:
            dpos = self.pos - body.pos
            pot -= body.mass / dpos.norm()
        return pot

    def get_ekin(self):
        """get the particle's kinetic energy"""
        self.ekin = 0.5 * self.mass * self.vel.square()
        return self.ekin

    def get_epot(self, bodies):
        """get the particle's potential energy"""
        self.epot = self.mass * self.get_pot(bodies)
        return self.epot

    def get_etot(self):
        """get the particle's total energy"""
        self.etot = self.ekin + self.epot
        return self.etot


# TODO: [pt_BR]: sobrescrever os métodos da classe 'Body' apropriadamente.
class BlackHole(Body):
    """A base class for black holes"""

    def __init__(self,
                 index=None,
                 nstep=None,
                 time=None,
                 tstep=None,
                 mass=None,
                 pos=None,
                 vel=None,
                 spin=None):

        Body.__init__(self, index, nstep, time, tstep, mass, pos, vel)
        if spin is None:
            self.spin = Vector(0, 0, 0)
        else:
            self.spin = spin

    def __repr__(self):
        fmt = '[{0:s}, {1:s}, {2:s}, {3:s}, {4:s}, {5:s}, {6:s}, {7:s}]'
        return fmt.format(repr(self.index), repr(self.nstep), repr(self.time),
                          repr(self.tstep), repr(self.mass), repr(self.pos),
                          repr(self.vel), repr(self.spin))


class Sph(Body):
    """A base class for sph particles"""
    pass


class Star(Body):
    """A base class for stars particles"""
    pass


class Universe(dict):
    """This class holds the particle types in the simulation"""

    def __new__(cls, *members):
        """constructor"""
        return dict.__new__(cls, members)

    def __init__(self, members=''):
        """initializer"""
        dict.__init__(self)
        if 'body' in members:
            self['body'] = [Body()]
        if 'bh' in members:
            self['bh'] = [BlackHole()]
        if 'star' in members:
            self['star'] = [Star()]
        if 'sph' in members:
            self['sph'] = [Sph()]


########## end of file ##########
