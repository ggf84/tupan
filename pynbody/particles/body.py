#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import math
import numpy as np
from collections import namedtuple
from pynbody.pbase import Pbase
try:
    from pynbody.lib import (p2p_acc_kernel, p2p_pot_kernel)
    HAVE_CL = True
except:
    HAVE_CL = False
    print('computing without OpenCL')


#HAVE_CL = False



class Bodies2(Pbase):
    """
    A base class for Body-type particles
    """
    def __init__(self):
        dtype = [('index', 'u8'), ('mass', 'f8'), ('eps2', 'f8'),   # eps2 -> radius
                 ('phi', 'f8'), ('stepdens', '2f8'), ('pos', '3f8'),
                 ('vel', '3f8'), ('acc', '3f8')]
        Pbase.__init__(self, dtype)


    def get_mass(self):
        return np.sum(self.mass)


    # Momentum methods

    def get_mass_mom(self):     # moment of inertia
        return np.sum(self.mass * (self.pos**2).sum(1))

    def get_linear_mom(self):
        return (self.mass * self.vel.T).sum(1)

    def get_angular_mom(self):
        return (self.mass * np.cross(self.pos, self.vel).T).sum(1)

    def get_angular_mom_squared(self):
        amom = self.get_angular_mom()
        return np.dot(amom, amom)


    # Center of Mass methods

    def get_rCoMass(self):
        return (self.mass * self.pos.T).sum(1) / self.get_mass()

    def get_vCoMass(self):
        return (self.mass * self.vel.T).sum(1) / self.get_mass()

    def reset_CoMass(self):
        self.pos -= self.get_rCoMass()
        self.vel -= self.get_vCoMass()


    # Energy methods

    def get_ekin(self):
        """get the bodies' total kinetic energy"""
        return 0.5 * np.sum(self.mass * (self.vel**2).sum(1))

    def get_epot(self):
        """get the bodies' total potential energy"""
        return 0.5 * np.sum(self.mass * self.phi)

    def get_energies(self):
        ekin = self.get_ekin()
        epot = self.get_epot()
        etot = ekin + epot
        Energy = namedtuple('Energy', ['kin', 'pot', 'tot'])
        energy = Energy(ekin, epot, etot)
        return energy


    # Evolve methods

    def drift(self, tau):
        self.pos += tau * self.vel

    def kick(self, tau):
        self.vel += tau * self.acc


    # Gravity methods

    def set_phi(self, objs):
        """
        set the all bodies' gravitational potential due to other bodies
        """
        def p2p_pot_pyrun(self, objs):
            _phi = []
            for bi in self:
                iphi = 0.0
                for bj in objs:
                    if bi['index'] != bj['index']:
                        dpos = bi['pos'] - bj['pos']
                        eps2 = bi['eps2'] + bj['eps2']
                        ds2 = np.dot(dpos, dpos) + eps2
                        iphi -= bj['mass'] / math.sqrt(ds2)
                _phi.append(iphi)
            return _phi

        if HAVE_CL:
            _phi = p2p_pot_kernel.run(self, objs)
        else:
            _phi = p2p_pot_pyrun(self, objs)
            _phi = np.asarray(_phi)

#        print(_phi)

        self.phi[:] = _phi


    def set_acc(self, objs):
        """set the all bodies' acceleration due to other bodies"""
        def p2p_acc_pyrun(self, objs):
            _acc = []
            for bi in self:
                iacc = np.zeros(4, dtype=np.float64)
                for bj in objs:
                    if bi['index'] != bj['index']:
                        dpos = bi['pos'] - bj['pos']
                        eps2 = bi['eps2'] + bj['eps2']
                        dvel = bi['vel'] - bj['vel']
                        M = bi['mass'] + bj['mass']

                        ds2 = np.dot(dpos, dpos) + eps2
                        dv2 = np.dot(dvel, dvel)

                        rinv = math.sqrt(1.0 / ds2)
                        r3inv = rinv * rinv * rinv

#                        iacc[3] += r3inv

                        e = 0.5*dv2 + M*rinv
                        iacc[3] += (e*e*e)/(M*M)

#                        iacc[3] -= rinv * bj['mass']

                        r3inv *= bj['mass']
                        iacc[0:3] -= r3inv * dpos
                _acc.append(iacc)
            return _acc

        if HAVE_CL:
            _acc = p2p_acc_kernel.run(self, objs)
#            elapsed = p2p_acc_kernel.run.selftimer.elapsed
#            gflops_count = p2p_acc_kernel.flops * len(self) * len(bodies) * 1.0e-9
#            print(' -- '*10)
#            print('Total kernel-run time: {0:g} s'.format(elapsed))
#            print('kernel-run Gflops/s: {0:g}'.format(gflops_count/elapsed))
        else:
            _acc = p2p_acc_pyrun(self, objs)
            _acc = np.asarray(_acc)

#        print('acc - '*10)
#        print(_acc)

        self.acc[:] = _acc[:,:-1]
        self.stepdens[:,0] = np.sqrt(_acc[:,-1]/len(objs))




















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
        def p2p_pot_pyrun(self, bodies):
            _pot = []
            for bi in self:
                ipot = 0.0
                for bj in bodies:
                    if bi['index'] != bj['index']:
                        dpos = bi['pos'] - bj['pos']
                        eps2 = bi['eps2'] + bj['eps2']
                        ds2 = np.dot(dpos, dpos) + eps2
                        ipot -= bj['mass'] / math.sqrt(ds2)
                _pot.append(ipot)
            return _pot

        if HAVE_CL:
            _pot = p2p_pot_kernel.run(self, bodies)
        else:
            _pot = p2p_pot_pyrun(self, bodies)
            _pot = np.asarray(_pot)

#        print(_pot)

        self.pot = _pot



    def calc_acc(self, bodies):
        """set the all bodies' acceleration due to other bodies"""
        def p2p_acc_pyrun(self, bodies):
            _acc = []
            for bi in self:
                iacc = np.zeros(4, dtype=np.float64)
                for bj in bodies:
                    if bi['index'] != bj['index']:
                        dpos = bi['pos'] - bj['pos']
                        eps2 = bi['eps2'] + bj['eps2']
                        dvel = bi['vel'] - bj['vel']
                        M = bi['mass'] + bj['mass']

                        ds2 = np.dot(dpos, dpos) + eps2
                        dv2 = np.dot(dvel, dvel)

                        rinv = math.sqrt(1.0 / ds2)
                        r3inv = rinv * rinv * rinv

#                        iacc[3] += r3inv

                        e = 0.5*dv2 + M*rinv
                        iacc[3] += (e*e*e)/(M*M)

#                        iacc[3] -= rinv * bj['mass']

                        r3inv *= bj['mass']
                        iacc[0:3] -= r3inv * dpos
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
            _acc = p2p_acc_pyrun(self, bodies)
            _acc = np.asarray(_acc)

#        print('acc - '*10)
#        print(_acc)

        self.acc = _acc[:,:-1]
        self.curr_step_density = np.sqrt(_acc[:,-1]/len(bodies))
#        self.curr_step_density = np.sqrt(_acc[:,-1])
#        self.curr_step_density = (_acc[:,-1]/len(bodies))**(1.0/3.0)


    def get_ekin(self):
        """get the bodies' total kinetic energy"""
        return 0.5 * np.sum(self.mass * (self.vel**2).T)

    def get_epot(self):
        """get the bodies' total potential energy"""
        return 0.5 * np.sum(self.mass * self.pot)

    def get_energies(self):
        ekin = self.get_ekin()
        epot = self.get_epot()
        etot = ekin + epot
        return (ekin, epot, etot)

    def get_rCoM(self):
        return (self.mass * self.pos.T).sum(1) / np.sum(self.mass)

    def set_rCoM(self, rshift):
        self.pos += rshift

    def get_vCoM(self):
        return (self.mass * self.vel.T).sum(1) / np.sum(self.mass)

    def set_vCoM(self, vshift):
        self.vel += vshift

    def reset_CoM(self):
        self.pos -= self.get_rCoM()
        self.vel -= self.get_vCoM()

    def get_linear_mom(self):
        return (self.mass * self.vel.T).sum(1)

    def get_angular_mom(self):
        return (self.mass * np.cross(self.pos, self.vel).T).sum(1)

    def drift(self, tau):
        self.pos += tau * self.vel

    def kick(self, tau):
        self.vel += tau * self.acc



########## end of file ##########
