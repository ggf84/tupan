#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This module holds base classes for particle types"""

import numpy as np
from .vector import Vector


class Body(object):
    """A base class for particles"""

    def __init__(self, fields=None):

        if fields:
            (self.index,
             self.nstep,
             self.tstep,
             self.time,
             self.mass,
             self.eps2,
             self.pot,
             self.pos,
             self.vel) = fields
        else:
            self.index = 0
            self.nstep = 0
            self.tstep = 0.0
            self.time = 0.0
            self.mass = 0.0
            self.eps2 = 0.0
            self.pot = 0.0
            self.pos = Vector(0.0, 0.0, 0.0)
            self.vel = Vector(0.0, 0.0, 0.0)

        self.acc = None
        self.ekin = None
        self.epot = None
        self.etot = None

    def __iter__(self):
        yield self.index
        yield self.nstep
        yield self.tstep
        yield self.time
        yield self.mass
        yield self.eps2
        yield self.pot
        yield self.pos
        yield self.vel

    def __repr__(self):
        fields = (self.index, self.nstep,
                  self.tstep, self.time,
                  self.mass, self.eps2, self.pot,
                  self.pos, self.vel)
        return '{fields}'.format(fields=fields)
#        return '{0}'.format(self[:])

    def set_acc(self, bodies):
        """set the particle's acceleration due to other bodies"""
        sumacc = Vector(0.0, 0.0, 0.0)
        for body in bodies:
            if self.index != body.index:
                dpos = self.pos - body.pos
                dposinv3 = dpos.square() ** (-1.5)
                sumacc -= body.mass * dpos * dposinv3
        self.acc = sumacc

    def get_acc(self):
        return self.acc

    def set_pot(self, bodies):
        """set the gravitational potential in the particle's location"""
        sumpot = 0.0
        for body in bodies:
            if self.index != body.index:
                dpos = self.pos - body.pos
                sumpot -= body.mass / dpos.norm()
        self.pot = sumpot

    def get_pot(self):
        return self.pot

    def get_ekin(self):
        """get the particle's kinetic energy"""
        self.ekin = 0.5 * self.mass * self.vel.square()
        return self.ekin

    def get_epot(self):
        """get the particle's potential energy"""
        self.epot = self.mass * self.get_pot()
        return self.epot

    def get_etot(self):
        """get the particle's total energy"""
        self.etot = self.ekin + self.epot
        return self.etot


# TODO: [pt_BR]: sobrescrever os métodos da classe 'Body' apropriadamente.
class BlackHole(Body):
    """A base class for black holes"""

    def __init__(self, fields=None):

        if fields:
            Body.__init__(self, fields[:-1])
            self.spin = list(fields[::-1][0])
        else:
            Body.__init__(self)
            self.spin = Vector(0.0, 0.0, 0.0)

    def __iter__(self):
        yield self.index
        yield self.nstep
        yield self.tstep
        yield self.time
        yield self.mass
        yield self.eps2
        yield self.pot
        yield self.pos
        yield self.vel
        yield self.spin

    def __repr__(self):
        fields = (self.index, self.nstep,
                  self.tstep, self.time,
                  self.mass, self.eps2, self.pot,
                  self.pos, self.vel, self.spin)
        return '{fields}'.format(fields=fields)
#        return '{0}'.format(self[:])


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
        self['body'] = {}
        self['body'].setdefault('array', np.array([], dtype=Body))
        self['body'].setdefault('format',
                                [('index', '<u8'), ('nstep', '<u8'),
                                 ('tstep', '<f8'), ('time', '<f8'),
                                 ('mass', '<f8'), ('eps2', '<f8'), ('pot', '<f8'),
                                 ('pos', '<f8', (3,)), ('vel', '<f8', (3,))])

        # BlackHole
        self['bh'] = {}
        self['bh'].setdefault('array', np.array([], dtype=BlackHole))
        self['bh'].setdefault('format',
                              [('index', '<u8'), ('nstep', '<u8'),
                               ('tstep', '<f8'), ('time', '<f8'),
                               ('mass', '<f8'), ('eps2', '<f8'), ('pot', '<f8'),
                               ('pos', '<f8', (3,)), ('vel', '<f8', (3,)),
                               ('spin', '<f8', (3,))])

        # Sph
        self['sph'] = {}
        self['sph'].setdefault('array', np.array([], dtype=Sph))
        self['sph'].setdefault('format', None)

        # Star
        self['star'] = {}
        self['star'].setdefault('array', np.array([], dtype=Star))
        self['star'].setdefault('format', None)

    def set_members(self, member='', data=None):
        if 'body' in member:
#            tmp = np.ndarray(len(data), dtype=Body)
#            tmp.setfield(data, dtype=Body)
#            self['body']['array'] = np.array(tmp, dtype=Body)
            self['body']['array'] = data

        if 'bh' in member:
#            tmp = np.ndarray(len(data), dtype=BlackHole)
#            tmp.setfield(data, dtype=BlackHole)
#            self['bh']['array'] = np.array(tmp, dtype=BlackHole)
            self['bh']['array'] = data

        if 'sph' in member:
            tmp = np.ndarray(len(data), dtype=Sph)
            tmp.setfield(data, dtype=Sph)
            self['sph']['array'] = np.array(tmp, dtype=Sph)
#            self['sph']['array'] = data

        if 'star' in member:
            tmp = np.ndarray(len(data), dtype=Star)
            tmp.setfield(data, dtype=Star)
            self['star']['array'] = np.array(tmp, dtype=Star)
#            self['star']['array'] = data


########## end of file ##########
