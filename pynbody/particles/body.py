#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""  """

from __future__ import print_function
import numpy as np
from math import sqrt
#from pynbody.vector import Vector
try:
    from pynbody.lib.kernels import (p2p_acc_kernel, p2p_pot_kernel)
    HAVE_CL = True
except:
    HAVE_CL = False
    print('computing without OpenCL')


#HAVE_CL = False


#class Body(object):
#    """A base class for a Body-type particle"""

#    def __init__(self, fields=None):

#        self.index = 0
#        self.nstep = 0
#        self.tstep = 0.0
#        self.time = 0.0
#        self.mass = 0.0
#        self.eps2 = 0.0
#        self.pot = 0.0
#        self.pos = Vector(0.0, 0.0, 0.0)
#        self.vel = Vector(0.0, 0.0, 0.0)

#        if fields:  # TODO: implementar usando self.__dict__['key'] = value
#            (self.index,
#             self.nstep,
#             self.tstep,
#             self.time,
#             self.mass,
#             self.eps2,
#             self.pot,
#             self.pos,
#             self.vel) = fields

#        self._acc = None
#        self._ekin = None
#        self._epot = None

#    def __repr__(self):
#        fields = (self.index, self.nstep,
#                  self.tstep, self.time,
#                  self.mass, self.eps2, self.pot,
#                  self.pos, self.vel)
#        return '{fields}'.format(fields=fields)

#    def __iter__(self):
#        yield self.index
#        yield self.nstep
#        yield self.tstep
#        yield self.time
#        yield self.mass
#        yield self.eps2
#        yield self.pot
#        yield self.pos
#        yield self.vel

#    def set_pot(self, bodies):
#        """set the body's gravitational potential due to other bodies"""
#        def py_set_pot(self, bodies):
#            _pot = 0.0
#            for bj in bodies:
#                if self.index != bj.index:
#                    dpos = self.pos - bj.pos
#                    eps2 = 0.5*(self.eps2 + bj.eps2)
#                    _pot -= bj.mass / dpos.smoothed_norm(eps2)
#            return _pot

#        if HAVE_CL:
#            _pot = cl_set_pot(self, bodies)
#        else:
#            _pot = py_set_pot(self, bodies)

#        self.pot = _pot

#    def get_ekin(self):
#        """get the body's kinetic energy"""
#        self._ekin = 0.5 * self.mass * self.vel.square()
#        return self._ekin

#    def get_epot(self):
#        """get the body's potential energy"""
#        self._epot = self.mass * self.pot
#        return self._epot

#    def get_etot(self):
#        """get the body's total energy"""
#        return self._ekin + self._epot

#    def set_acc(self, bodies):
#        """set the body's acceleration due to other bodies"""
#        def py_set_acc(self, bodies):
#            _acc = Vector(0.0, 0.0, 0.0)
#            for body in bodies:
#                if self.index != body.index:
#                    dpos = self.pos - body.pos
#                    eps2 = 0.5*(self.eps2 + body.eps2)
#                    dposinv3 = dpos.smoothed_square(eps2) ** (-1.5)
#                    _acc -= body.mass * dpos * dposinv3
#            return _acc

#        if HAVE_CL:
#            raise NotImplementedError('NotImplemented')
#            _acc = cl_set_acc(self, bodies)
#        else:
#            _acc = py_set_acc(self, bodies)

#        self._acc = _acc

#    def get_acc(self):
#        """get the body's acceleration"""
#        return self._acc

















class Bodies(object):
    """A base class for all Body-type particles"""

    def __init__(self):
        dtype = [('index', 'u8'), ('nstep', 'u8'),
                 ('tstep', 'f8'), ('time', 'f8'),
                 ('mass', 'f8'), ('eps2', 'f8'), ('pot', 'f8'),
                 ('pos', 'f8', (3,)), ('vel', 'f8', (3,))]
        self._array = np.array([], dtype=dtype)

        # total mass
        self._total_mass = 0.0

    def __repr__(self):
        return '{array}'.format(array=self._array)

    def __iter__(self):
        return iter(self._array)

    def __reversed__(self):
        return reversed(self._array)

    def __len__(self):
        return self._array.size

    def __getitem__(self, index):
        b = Bodies()
        b.fromlist(self._array[index])
        return b

    def get_data(self):
        return self._array

    def fromlist(self, data):
        self._array = np.asarray(data, dtype=self._array.dtype)
        self._total_mass = np.sum(self._array['mass'])

    def insert_body(self, index, body):
        """Inserts a new body before the given index, updating the total mass"""
        self._total_mass += body.mass
        self._array = np.insert(self._array, index, tuple(body))

    def remove_body(self, bindex):
        """Remove the body of index 'bindex' and update the total mass"""
        arr = self._array
        if arr.size < 1:
            return 'there is no more bodies to remove!'
        self._total_mass -= arr[np.where(arr['index'] == bindex)]['mass'][0]
        self._array = arr[np.where(arr['index'] != bindex)]

    # properties

    def _get_index(self):
        return self._array['index']
    def _set_index(self, _index):
        self._array['index'] = _index
    index = property(_get_index, _set_index)

    def _get_nstep(self):
        return self._array['nstep']
    def _set_nstep(self, _nstep):
        self._array['nstep'] = _nstep
    nstep = property(_get_nstep, _set_nstep)

    def _get_tstep(self):
        return self._array['tstep']
    def _set_tstep(self, _tstep):
        self._array['tstep'] = _tstep
    tstep = property(_get_tstep, _set_tstep)

    def _get_time(self):
        return self._array['time']
    def _set_time(self, _time):
        self._array['time'] = _time
    time = property(_get_time, _set_time)

    def _get_mass(self):
        return self._array['mass']
    def _set_mass(self, _mass):
        self._array['mass'] = _mass
    mass = property(_get_mass, _set_mass)

    def _get_eps2(self):
        return self._array['eps2']
    def _set_eps2(self, _eps2):
        self._array['eps2'] = _eps2
    eps2 = property(_get_eps2, _set_eps2)

    def _get_pot(self):
        return self._array['pot']
    def _set_pot(self, _pot):
        self._array['pot'] = _pot
    pot = property(_get_pot, _set_pot)

    def _get_pos(self):
        return self._array['pos']
    def _set_pos(self, _pos):
        self._array['pos'] = _pos
    pos = property(_get_pos, _set_pos)

    def _get_vel(self):
        return self._array['vel']
    def _set_vel(self, _vel):
        self._array['vel'] = _vel
    vel = property(_get_vel, _set_vel)


    def get_total_mass(self):
        return self._total_mass
    def set_total_mass(self, mtot):
        self._total_mass = mtot


    def calc_pot(self, bodies):
        """set the all bodies' gravitational potential due to other bodies"""
        def py_pot_perform_calc(self, bodies):
            _pot = []
            for bi in self._array:
                ipot = 0.0
                for bj in bodies:
                    if bi['index'] != bj['index']:
                        dpos = bi['pos'] - bj['pos']
                        eps2 = 0.5*(bi['eps2'] + bj['eps2'])
                        ds2 = np.dot(dpos, dpos) + eps2
                        ipot -= bj['mass'] / sqrt(ds2)
                _pot.append(ipot)
            return _pot

        if HAVE_CL:
            _pot = p2p_pot_kernel.run(self, bodies)
        else:
            _pot = py_pot_perform_calc(self, bodies)
            _pot = np.asarray(_pot)

        print(_pot)

        self._array['pot'] = _pot


    def calc_acc(self, bodies):
        """set the all bodies' acceleration due to other bodies"""
        def py_acc_perform_calc(self, bodies):
            _acc = []
            for bi in self._array:
                iacc = np.zeros(4, dtype=np.float64)
                for bj in bodies:
                    if bi['index'] != bj['index']:
                        dpos = bi['pos'] - bj['pos']
                        eps2 = 0.5*(bi['eps2'] + bj['eps2'])
                        ds2 = np.dot(dpos, dpos) + eps2
                        r2inv = 1.0 / ds2
                        rinv = sqrt(r2inv)
                        mrinv = bj['mass'] * rinv
                        mr3inv = mrinv * r2inv
                        iacc[3] -= mrinv
                        iacc[0:3] -= mr3inv * dpos
                _acc.append(iacc)
            return _acc

        if HAVE_CL:
            _acc = p2p_acc_kernel.run(self, bodies)
            print('************* ', p2p_acc_kernel.run.selftimer.elapsed)
        else:
            _acc = py_acc_perform_calc(self, bodies)
            _acc = np.asarray(_acc)

        print('acc '*20)
        print(self._array['pos'])
        print('-'*25)
        print(_acc)

#        self._array['pot'] = _pot


    def get_ekin(self):
        """get the bodies' total kinetic energy"""
        return 0.5 * np.sum(self._array['mass'] * (self._array['vel']**2).T)

    def get_epot(self):
        """get the bodies' total potential energy"""
        return np.sum(self._array['mass'] * self._array['pot'])















# original

#class Bodies_2(list):
#    """A base class for all Body-type particles"""

#    def __new__(cls):
#        """constructor"""
#        return list.__new__(cls)

#    def __init__(self):
#        list.__init__(self)

#        # total mass
#        self._total_mass = 0.0

#    def append_member(self, body):
#        """Append a new member for bodies' list and update the total mass"""
#        self._total_mass += body.mass
#        self.append(body)

#    def pop_member(self, body_index):
#        """
#        Find the index from body-index and remove and return the member at this
#        index, updating the total mass.
#        """
#        cmpidx = lambda s: s.index == body_index
#        list_index = self.index(filter(cmpidx, self)[0])
#        self._total_mass -= self[list_index].mass
#        return self.pop(list_index)

#    def select_attribute(self, attr):
#        return [getattr(b, attr) for b in self]

#    def get_total_mass(self):
#        return self._total_mass

#    def set_total_mass(self, mtot):
#        self._total_mass = mtot

#    def set_pot(self, bodies):
#        """set the all bodies' gravitational potential due to other bodies"""
#        def py_set_pot(self, bodies):
#            _pot = []
#            for bi in self:
#                ipot = 0.0
#                for bj in bodies:
#                    if bi.index != bj.index:
#                        dpos = bi.pos - bj.pos
#                        eps2 = 0.5*(bi.eps2 + bj.eps2)
#                        ipot -= bj.mass / dpos.smoothed_norm(eps2)
#                _pot.append(ipot)
#            return _pot

#        if HAVE_CL:
#            _pot = cl_set_pot(self, bodies)
#        else:
#            _pot = py_set_pot(self, bodies)

#        for (b, p) in zip(self, _pot):
#            b.pot = p

#    def get_ekin(self):
#        """get the bodies' total kinetic energy"""
#        return 0.5 * reduce(lambda x, y: x + y, 
#                            [b.mass * b.vel.square() for b in self])

#    def get_epot(self):
#        """get the bodies' total potential energy"""
#        return reduce(lambda x, y: x + y, [b.mass * b.pot for b in self])

#    def set_acc(self, bodies):
#        """set the all bodies' acceleration due to other bodies"""
#        def py_set_acc(self, bodies):
#            _acc = []
#            for bi in self:
#                iacc = Vector(0.0, 0.0, 0.0)
#                for bj in bodies:
#                    if bi.index != bj.index:
#                        dpos = bi.pos - bj.pos
#                        eps2 = 0.5*(bi.eps2 + bj.eps2)
#                        dposinv3 = dpos.smoothed_square(eps2) ** (-1.5)
#                        iacc -= bj.mass * dpos * dposinv3
#                _acc.append(iacc)
#            return _acc

#        if HAVE_CL:
#            raise NotImplementedError('cl_set_acc NotImplemented')
#            _acc = cl_set_acc(self, bodies)
#        else:
#            _acc = py_set_acc(self, bodies)

#        for (b, a) in zip(self, _acc):
#            b._acc = a


########## end of file ##########
