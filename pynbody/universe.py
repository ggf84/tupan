#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""  """

from vector import Vector


class Body(object):
    """A base class for particles"""

    def __init__(self, index=0, mass=0.0,
                 pos=Vector(0, 0, 0), vel=Vector(0, 0, 0)):
        """initializer"""
        self.index = index
        self.mass = mass
        self.pos = pos
        self.vel = vel

        self.acc = None
        self.ekin = None
        self.epot = None
        self.etot = None

    def __repr__(self):
        fmt = '[{0:s}, {1:s}, {2:s}, {3:s}]'
        return fmt.format(repr(self.index), repr(self.mass),
                          repr(self.pos), repr(self.vel))

    def get_acc(self, bodies):
        """get the particle's acceleration due to other bodies"""
        self.acc = Vector(0, 0, 0)
        for body in bodies:
            dr = self.pos - body.pos
            drinv3 = dr.square() ** (-1.5)
            self.acc -= body.mass * dr * drinv3
        return self.acc

    def get_pot(self, bodies):
        """get the gravitational potential in the particle's location"""
        pot = 0.0
        for body in bodies:
            dr = self.pos - body.pos
            pot -= body.mass / dr.norm()
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


# TODO: [pt_BR]: sobrescrever os métodos da classe 'Particle' apropriadamente.
class BlackHole(Body):
    """A base class for black holes"""

    def __init__(self, index=0, mass=0.0, pos=Vector(0, 0, 0),
                 vel=Vector(0, 0, 0), spin=Vector(0, 0, 0)):
        """initializer"""
        Body.__init__(self, index, mass, pos, vel)
        self.spin = spin

    def __repr__(self):
        fmt = '[{0:s}, {1:s}, {2:s}, {3:s}, {4:s}]'
        return fmt.format(repr(self.index), repr(self.mass), repr(self.pos),
                          repr(self.vel), repr(self.spin))


class SPH(Body):
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
        for member in members:
            self[member] = None


########## end of file ##########
