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
        self.index = np.array([], dtype='u8')
        self.step_number = np.array([], dtype='u8')
        self.curr_step_density = np.array([], dtype='f8')
        self.next_step_density = np.array([], dtype='f8')
        self.time = np.array([], dtype='f8')
        self.tstep = np.array([], dtype='f8')
        self.mass = np.array([], dtype='f8')
        self.eps2 = np.array([], dtype='f8')
        self.pot = np.array([], dtype='f8')
        self.pos = np.array([], dtype='f8')
        self.vel = np.array([], dtype='f8')
        self.acc = np.array([], dtype='f8')

        self.dtype = [('index', 'u8'), ('step_number', 'u8'),
                      ('curr_step_density', 'f8'), ('next_step_density', 'f8'),
                      ('time', 'f8'), ('tstep', 'f8'),
                      ('mass', 'f8'), ('eps2', 'f8'), ('pot', 'f8'),
                      ('pos', '3f8'), ('vel', '3f8'), ('acc', '3f8')]

        # total mass
        self.total_mass = 0.0

    def from_cmpd_struct(self, _array):
        for attr in dict(self.dtype).iterkeys():
            setattr(self, attr, _array[attr])

    def to_cmpd_struct(self):
        _array = np.empty(len(self), dtype=self.dtype)
        for attr in dict(self.dtype).iterkeys():
            _array[attr] = getattr(self, attr)
        return _array#.view(np.recarray)

    def concatenate(self, obj1):
        new_obj = Bodies()
        for attr in dict(self.dtype).iterkeys():
            obj0attr = getattr(self, attr)
            obj1attr = getattr(obj1, attr)
            setattr(new_obj, attr, np.concatenate((obj0attr, obj1attr)))
        return new_obj

    def __repr__(self):
        return '{array}'.format(array=self.to_cmpd_struct())

    def __iter__(self):     # XXX:
        return iter(self.to_cmpd_struct())

    def __len__(self):
        return len(self.index)

    def __reversed__(self):     # XXX:
        return reversed(self.to_cmpd_struct())

    def __getitem__(self, index):     # XXX:
        s = self.to_cmpd_struct()[index]
        if not isinstance(s, np.ndarray):
            s = np.asarray([s], dtype=self.dtype)
        obj = Bodies()
        obj.from_cmpd_struct(s)
        return obj

    def fromlist(self, data):
        self.from_cmpd_struct(np.asarray(data, dtype=self.dtype))
        self.total_mass = np.sum(self.mass)

    def insert_body(self, index, body):     # XXX:
        """Inserts a new body before the given index, updating the total mass"""
        self.total_mass += body.mass
        self.array = np.insert(self.array, index, tuple(body))

    def remove_body(self, bindex):     # XXX:
        """Remove the body of index 'bindex' and update the total mass"""
        arr = self.array
        if arr.size < 1:
            return 'there is no more bodies to remove!'
        self.total_mass -= arr[np.where(arr['index'] == bindex)]['mass'][0]
        self.array = arr[np.where(arr['index'] != bindex)]


    def calc_pot(self, bodies):
        """set the all bodies' gravitational potential due to other bodies"""
        def py_pot_perform_calc(self, bodies):
            _pot = []
            for bi in self:
                ipot = 0.0
                for bj in bodies:
                    if bi.index != bj.index:
                        dpos = bi.pos - bj.pos
                        eps2 = 0.5*(bi.eps2 + bj.eps2)
                        ds2 = np.dot(dpos, dpos) + eps2
                        ipot -= bj.mass / math.sqrt(ds2)
                _pot.append(ipot)
            return _pot

        if HAVE_CL:
            _pot = p2p_pot_kernel.run(self, bodies)
        else:
            _pot = py_pot_perform_calc(self, bodies)
            _pot = np.asarray(_pot)

#        print(_pot)

        self.pot = _pot



    def calc_acc(self, bodies):
        """set the all bodies' acceleration due to other bodies"""
        def py_acc_perform_calc(self, bodies):
            _acc = []
            for bi in self:
                iacc = np.zeros(4, dtype=np.float64)
                for bj in bodies:
                    if bi.index != bj.index:
                        dpos = bi.pos - bj.pos
                        eps2 = 0.5*(bi.eps2 + bj.eps2)
                        ds2 = np.dot(dpos, dpos) + eps2
                        r2inv = 1.0 / ds2
                        rinv = math.sqrt(r2inv)
                        mrinv = bj.mass * rinv
                        mr3inv = mrinv * r2inv
                        iacc[3] -= mrinv
                        iacc[0:3] -= mr3inv * dpos
                _acc.append(iacc)
            return _acc

        if HAVE_CL:
            _acc = p2p_acc_kernel.run(self, bodies)
#            elapsed = p2p_acc_kernel.run.selftimer.elapsed
#            gflops_count = p2p_acc_kernel.flops * len(self) * len(bodies) * 1.0e-9
#            print(' -- '*10)
#            print('Total kernel-run time: {0:g} s'.format(elapsed))
#            print('kernel-run Gflops/s: {0:g}'.format(gflops_count/elapsed))
        else:
            _acc = py_acc_perform_calc(self, bodies)
            _acc = np.asarray(_acc)

#        print('acc - '*10)
#        print(_acc)

        self.acc = _acc[:,:-1]
        self.curr_step_density = np.sqrt(_acc[:,-1]/len(bodies))
#        self.curr_step_density = (_acc[:,-1]/len(bodies))**(1.0/3.0)


    def get_ekin(self):
        """get the bodies' total kinetic energy"""
        return 0.5 * np.sum(self.mass * (self.vel**2).T)

    def get_epot(self):
        """get the bodies' total potential energy"""
        return 0.5 * np.sum(self.mass * self.pot)

    def get_CoM(self):
        return (self.mass * self.pos.T).sum(1) / np.sum(self.mass)

    def get_linear_mom(self):
        return (self.mass * self.vel.T).sum(1)

    def get_angular_mom(self):
        return (self.mass * np.cross(self.pos, self.vel).T).sum(1)

    def drift(self, tau):
#        self.pos += (self.tstep * self.vel.T).T
        self.pos += tau * self.vel

    def kick(self, tau):
#        self.vel += (self.tstep * self.acc.T).T
        self.vel += tau * self.acc































#class Bodies2(object):
#    """A base class for Body-type particles"""

#    def __init__(self):
#        self.array = np.array([],
#                              dtype=[('index', 'u8'), ('nstep', 'u8'),
#                                     ('step_density', 'f8'), ('time', 'f8'),
#                                     ('mass', 'f8'), ('eps2', 'f8'), ('pot', 'f8'),
#                                     ('pos', 'f8', (3,)), ('vel', 'f8', (3,)),
#                                     ('acc', 'f8', (3,))])

#        # total mass
#        self._total_mass = 0.0

#    def __repr__(self):
#        return '{array}'.format(array=self.array)

#    def __iter__(self):
#        return iter(self.array)

#    def __reversed__(self):
#        return reversed(self.array)

#    def __len__(self):
#        return len(self.array)

#    def __getitem__(self, index):
#        b = Bodies()
#        b.fromlist(self.array[index])
#        return b

#    def get_data(self):
#        return self.array

#    def fromlist(self, data):
#        self.array = np.asarray(data, dtype=self.array.dtype)
#        self._total_mass = np.sum(self.array['mass'])

#    def insert_body(self, index, body):
#        """Inserts a new body before the given index, updating the total mass"""
#        self._total_mass += body.mass
#        self.array = np.insert(self.array, index, tuple(body))

#    def remove_body(self, bindex):
#        """Remove the body of index 'bindex' and update the total mass"""
#        arr = self.array
#        if arr.size < 1:
#            return 'there is no more bodies to remove!'
#        self._total_mass -= arr[np.where(arr['index'] == bindex)]['mass'][0]
#        self.array = arr[np.where(arr['index'] != bindex)]

#    # properties

#    def _get_index(self):
#        return self.array['index']
#    def _set_index(self, _index):
#        self.array['index'] = _index
#    index = property(_get_index, _set_index)

#    def _get_nstep(self):
#        return self.array['nstep']
#    def _set_nstep(self, _nstep):
#        self.array['nstep'] = _nstep
#    nstep = property(_get_nstep, _set_nstep)

#    def _get_step_density(self):
#        return self.array['step_density']
#    def _set_step_density(self, _step_density):
#        self.array['step_density'] = _step_density
#    step_density = property(_get_step_density, _set_step_density)

#    def _get_time(self):
#        return self.array['time']
#    def _set_time(self, _time):
#        self.array['time'] = _time
#    time = property(_get_time, _set_time)

#    def _get_mass(self):
#        return self.array['mass']
#    def _set_mass(self, _mass):
#        self.array['mass'] = _mass
#    mass = property(_get_mass, _set_mass)

#    def _get_eps2(self):
#        return self.array['eps2']
#    def _set_eps2(self, _eps2):
#        self.array['eps2'] = _eps2
#    eps2 = property(_get_eps2, _set_eps2)

#    def _get_pot(self):
#        return self.array['pot']
#    def _set_pot(self, _pot):
#        self.array['pot'] = _pot
#    pot = property(_get_pot, _set_pot)

#    def _get_pos(self):
#        return self.array['pos']
#    def _set_pos(self, _pos):
#        self.array['pos'] = _pos
#    pos = property(_get_pos, _set_pos)

#    def _get_vel(self):
#        return self.array['vel']
#    def _set_vel(self, _vel):
#        self.array['vel'] = _vel
#    vel = property(_get_vel, _set_vel)


#    def get_total_mass(self):
#        return self._total_mass
#    def set_total_mass(self, mtot):
#        self._total_mass = mtot


#    def calc_pot(self, bodies):
#        """set the all bodies' gravitational potential due to other bodies"""
#        def py_pot_perform_calc(self, bodies):
#            _pot = []
#            for bi in self.array:
#                ipot = 0.0
#                for bj in bodies:
#                    if bi['index'] != bj['index']:
#                        dpos = bi['pos'] - bj['pos']
#                        eps2 = 0.5*(bi['eps2'] + bj['eps2'])
#                        ds2 = np.dot(dpos, dpos) + eps2
#                        ipot -= bj['mass'] / math.sqrt(ds2)
#                _pot.append(ipot)
#            return _pot

#        if HAVE_CL:
#            _pot = p2p_pot_kernel.run(self, bodies)
#        else:
#            _pot = py_pot_perform_calc(self, bodies)
#            _pot = np.asarray(_pot)

#        print(_pot)

#        self.array['pot'] = _pot


#    def calc_acc(self, bodies):
#        """set the all bodies' acceleration due to other bodies"""
#        def py_acc_perform_calc(self, bodies):
#            _acc = []
#            for bi in self.array:
#                iacc = np.zeros(4, dtype=np.float64)
#                for bj in bodies:
#                    if bi['index'] != bj['index']:
#                        dpos = bi['pos'] - bj['pos']
#                        eps2 = 0.5*(bi['eps2'] + bj['eps2'])
#                        ds2 = np.dot(dpos, dpos) + eps2
#                        r2inv = 1.0 / ds2
#                        rinv = math.sqrt(r2inv)
#                        mrinv = bj['mass'] * rinv
#                        mr3inv = mrinv * r2inv
#                        iacc[3] -= mrinv
#                        iacc[0:3] -= mr3inv * dpos
#                _acc.append(iacc)
#            return _acc

#        if HAVE_CL:
#            _acc = p2p_acc_kernel.run(self, bodies)
#            elapsed = p2p_acc_kernel.run.selftimer.elapsed
#            gflops_count = p2p_acc_kernel.flops * len(self) * len(bodies) * 1.0e-9
#            print(' -- '*10)
#            print('Total kernel-run time: {0:g} s'.format(elapsed))
#            print('kernel-run Gflops/s: {0:g}'.format(gflops_count/elapsed))
#        else:
#            _acc = py_acc_perform_calc(self, bodies)
#            _acc = np.asarray(_acc)

#        print('acc - '*10)
#        print(_acc)

#        self.array['acc'] = _acc[:,:-1]
#        self.array['step_density'] = -_acc[:,-1]

#    def get_ekin(self):
#        """get the bodies' total kinetic energy"""
#        return 0.5 * np.sum(self.array['mass'] * (self.array['vel']**2).T)

#    def get_epot(self):
#        """get the bodies' total potential energy"""
#        return np.sum(self.array['mass'] * self.array['pot'])


#    def drift(self, tau):
#        self.array['pos'] += tau * self.array['vel']

#    def kick(self, tau):
#        self.array['vel'] += tau * self.array['acc']

















########## end of file ##########
