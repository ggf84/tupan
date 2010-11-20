#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This module holds base classes for particle types"""

import sys
from vector import Vector


class Body(object):
    """A base class for particles"""

    def __init__(self,
                 nstep=None,
                 tstep=None,
                 time=None,
                 mass=None,
                 pos=None,
                 vel=None):

        if nstep is None:
            self.nstep = 0
        else:
            self.nstep = nstep

        if tstep is None:
            self.tstep = 0.0
        else:
            self.tstep = tstep

        if time is None:
            self.time = 0.0
        else:
            self.time = time

        if mass is None:
            self.mass = 0.0
        else:
            self.mass = mass

        if pos is None:
            self.pos = Vector(0.0, 0.0, 0.0)
        else:
            self.pos = pos

        if vel is None:
            self.vel = Vector(0.0, 0.0, 0.0)
        else:
            self.vel = vel

        self.ekin = None
        self.epot = None
        self.etot = None

    def __repr__(self):
        fmt = '({0:s}, {1:s}, {2:s}, {3:s}, {4:s}, {5:s})'
        return fmt.format(repr(self.nstep), repr(self.tstep),
                          repr(self.time), repr(self.mass),
                          repr(self.pos), repr(self.vel))

    def __getitem__(self, index):
        return (self.nstep, self.tstep, self.time, self.mass, self.pos, self.vel)[index]

    def __iter__(self):
        yield self.nstep
        yield self.tstep
        yield self.time
        yield self.mass
        yield self.pos
        yield self.vel

    def get_acc(self, bodies):
        """get the particle's acceleration due to other bodies"""
        acc = Vector(0.0, 0.0, 0.0)
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
                 nstep=None,
                 tstep=None,
                 time=None,
                 mass=None,
                 pos=None,
                 vel=None,
                 spin=None):

        Body.__init__(self, nstep, tstep, time, mass, pos, vel)
        if spin is None:
            self.spin = Vector(0.0, 0.0, 0.0)
        else:
            self.spin = spin

    def __repr__(self):
        fmt = '({0:s}, {1:s}, {2:s}, {3:s}, {4:s}, {5:s}, {6:s})'
        return fmt.format(repr(self.nstep), repr(self.tstep),
                          repr(self.time), repr(self.mass),
                          repr(self.pos), repr(self.vel), repr(self.spin))

    def __getitem__(self, index):
        return (self.nstep, self.tstep, self.time, self.mass, self.pos, self.vel, self.spin)[index]

    def __iter__(self):
        yield self.nstep
        yield self.tstep
        yield self.time
        yield self.mass
        yield self.pos
        yield self.vel
        yield self.spin

class Sph(Body):
    """A base class for sph particles"""
    pass


class Star(Body):
    """A base class for stars particles"""
    pass


class Universe(dict):
    """This class holds the particle types in the simulation"""

    def __new__(cls):
        """constructor"""
        return dict.__new__(cls)

    def __init__(self):
        """initializer"""
        dict.__init__(self)

        # Body
        self['body'] = {'dtype': [('nstep', 'u4'), ('tstep', 'f8'),
                                  ('time', 'f8'), ('mass', 'f8'),
                                  ('pos', 'f8', (3,)), ('vel', 'f8', (3,))]}

        # BlackHole
        self['bh'] = {'dtype': [('nstep', 'u4'), ('tstep', 'f8'),
                                ('time', 'f8'), ('mass', 'f8'),
                                ('pos', 'f8', (3,)), ('vel', 'f8', (3,)),
                                ('spin', 'f8', (3,))]}

        # Sph
        self['sph'] = {'dtype': None}

        # Star
        self['star'] = {'dtype': None}

    def get_member(self, member='', data=None,
                   body_idx = iter(xrange(sys.maxint)),
                   bh_idx = iter(xrange(sys.maxint)),
                   sph_idx = iter(xrange(sys.maxint)),
                   star_idx = iter(xrange(sys.maxint))):

        if 'body' in member:
            try:
                if not isinstance(data, Body):
                    raise TypeError('expcted a \'Body\' type')
                self['body'][body_idx.next()] = data
            except TypeError:
                raise

        if 'bh' in member:
            try:
                if not isinstance(data, BlackHole):
                    raise TypeError('expcted a \'BlackHole\' type')
                self['bh'][bh_idx.next()] = data
            except TypeError:
                raise

        if 'sph' in member:
            try:
                if not isinstance(data, Sph):
                    raise TypeError('expcted a \'Sph\' type')
                self['sph'][sph_idx.next()] = data
            except TypeError:
                raise

        if 'star' in member:
            try:
                if not isinstance(data, Star):
                    raise TypeError('expcted a \'Star\' type')
                self['star'][star_idx.next()] = data
            except TypeError:
                raise


########## end of file ##########
