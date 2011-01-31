#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import math
import numpy as np
try:
    from pynbody.lib.kernels import (p2p_acc_kernel, p2p_pot_kernel)
    HAVE_CL = True
except:
    HAVE_CL = False
    print('computing without OpenCL')


#HAVE_CL = False


class Bodies(object):
    """A base class for Body-type particles"""

    def __init__(self):
        self.array = np.array([],
                              dtype=[('index', 'u8'), ('nstep', 'u8'),
                                     ('tstep', 'f8'), ('time', 'f8'),
                                     ('mass', 'f8'), ('eps2', 'f8'), ('pot', 'f8'),
                                     ('pos', 'f8', (3,)), ('vel', 'f8', (3,))])

        # total mass
        self._total_mass = 0.0

    def __repr__(self):
        return '{array}'.format(array=self.array)

    def __iter__(self):
        return iter(self.array)

    def __reversed__(self):
        return reversed(self.array)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        b = Bodies()
        b.fromlist(self.array[index])
        return b

    def get_data(self):
        return self.array

    def fromlist(self, data):
        self.array = np.asarray(data, dtype=self.array.dtype)
        self._total_mass = np.sum(self.array['mass'])

    def insert_body(self, index, body):
        """Inserts a new body before the given index, updating the total mass"""
        self._total_mass += body.mass
        self.array = np.insert(self.array, index, tuple(body))

    def remove_body(self, bindex):
        """Remove the body of index 'bindex' and update the total mass"""
        arr = self.array
        if arr.size < 1:
            return 'there is no more bodies to remove!'
        self._total_mass -= arr[np.where(arr['index'] == bindex)]['mass'][0]
        self.array = arr[np.where(arr['index'] != bindex)]

    # properties

    def _get_index(self):
        return self.array['index']
    def _set_index(self, _index):
        self.array['index'] = _index
    index = property(_get_index, _set_index)

    def _get_nstep(self):
        return self.array['nstep']
    def _set_nstep(self, _nstep):
        self.array['nstep'] = _nstep
    nstep = property(_get_nstep, _set_nstep)

    def _get_tstep(self):
        return self.array['tstep']
    def _set_tstep(self, _tstep):
        self.array['tstep'] = _tstep
    tstep = property(_get_tstep, _set_tstep)

    def _get_time(self):
        return self.array['time']
    def _set_time(self, _time):
        self.array['time'] = _time
    time = property(_get_time, _set_time)

    def _get_mass(self):
        return self.array['mass']
    def _set_mass(self, _mass):
        self.array['mass'] = _mass
    mass = property(_get_mass, _set_mass)

    def _get_eps2(self):
        return self.array['eps2']
    def _set_eps2(self, _eps2):
        self.array['eps2'] = _eps2
    eps2 = property(_get_eps2, _set_eps2)

    def _get_pot(self):
        return self.array['pot']
    def _set_pot(self, _pot):
        self.array['pot'] = _pot
    pot = property(_get_pot, _set_pot)

    def _get_pos(self):
        return self.array['pos']
    def _set_pos(self, _pos):
        self.array['pos'] = _pos
    pos = property(_get_pos, _set_pos)

    def _get_vel(self):
        return self.array['vel']
    def _set_vel(self, _vel):
        self.array['vel'] = _vel
    vel = property(_get_vel, _set_vel)


    def get_total_mass(self):
        return self._total_mass
    def set_total_mass(self, mtot):
        self._total_mass = mtot


    def calc_pot(self, bodies):
        """set the all bodies' gravitational potential due to other bodies"""
        def py_pot_perform_calc(self, bodies):
            _pot = []
            for bi in self.array:
                ipot = 0.0
                for bj in bodies:
                    if bi['index'] != bj['index']:
                        dpos = bi['pos'] - bj['pos']
                        eps2 = 0.5*(bi['eps2'] + bj['eps2'])
                        ds2 = np.dot(dpos, dpos) + eps2
                        ipot -= bj['mass'] / math.sqrt(ds2)
                _pot.append(ipot)
            return _pot

        if HAVE_CL:
            _pot = p2p_pot_kernel.run(self, bodies)
        else:
            _pot = py_pot_perform_calc(self, bodies)
            _pot = np.asarray(_pot)

        print(_pot)

        self.array['pot'] = _pot


    def calc_acc(self, bodies):
        """set the all bodies' acceleration due to other bodies"""
        def py_acc_perform_calc(self, bodies):
            _acc = []
            for bi in self.array:
                iacc = np.zeros(4, dtype=np.float64)
                for bj in bodies:
                    if bi['index'] != bj['index']:
                        dpos = bi['pos'] - bj['pos']
                        eps2 = 0.5*(bi['eps2'] + bj['eps2'])
                        ds2 = np.dot(dpos, dpos) + eps2
                        r2inv = 1.0 / ds2
                        rinv = math.sqrt(r2inv)
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
        print(self.array['pos'])
        print('-'*25)
        print(_acc)

#        self.array['pot'] = _pot


    def get_ekin(self):
        """get the bodies' total kinetic energy"""
        return 0.5 * np.sum(self.array['mass'] * (self.array['vel']**2).T)

    def get_epot(self):
        """get the bodies' total potential energy"""
        return np.sum(self.array['mass'] * self.array['pot'])















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
