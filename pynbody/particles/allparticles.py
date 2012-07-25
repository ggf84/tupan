#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import copy
import numpy as np
from .sph import Sph
from .body import Body
from .blackhole import BlackHole
from ..lib import gravity
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Particles"]


ALL_PARTICLE_TYPES = ["sph", "body", "blackhole"]


def make_common_attrs(cls):
    def make_property(attr, doc):
        def fget(self):
            seq = [getattr(obj, attr) for obj in self.values() if obj.n]
            if len(seq) == 1:
                return seq[0]
            if len(seq) > 1:
                return np.concatenate(seq)
            return np.concatenate([getattr(obj, attr) for obj in self.values()])
        def fset(self, value):
            for obj in self.values():
                if obj.n:
                    try:
                        setattr(obj, attr, value[:obj.n])
                        value = value[obj.n:]
                    except:
                        setattr(obj, attr, value)
        def fdel(self):
            raise NotImplementedError()
        return property(fget, fset, fdel, doc)
    attrs = ((i[0], cls.__name__+'\'s '+i[2]) for i in cls.attributes)
    for (attr, doc) in attrs:
        setattr(cls, attr, make_property(attr, doc))
    return cls


@decallmethods(timings)
@make_common_attrs
class Particles(dict):
    """
    This class holds the particle types in the simulation.
    """
    attributes = [# common attributes
                  ('id', 'u8', 'index'),
                  ('mass', 'f8', 'mass'),
                  ('pos', '3f8', 'position'),
                  ('vel', '3f8', 'velocity'),
                  ('acc', '3f8', 'acceleration'),
                  ('phi', 'f8', 'potential'),
                  ('eps2', 'f8', 'softening'),
                  ('t_curr', 'f8', 'current time'),
                  ('dt_prev', 'f8', 'previous time-step'),
                  ('dt_next', 'f8', 'next time-step'),
                  ('nstep', 'u8', 'step number'),
                 ]

    def __init__(self, nstar=0, nbh=0, nsph=0):
        """
        Initializer
        """
        super(Particles, self).__init__(
                                        sph=Sph(nsph),
                                        body=Body(nstar),
                                        blackhole=BlackHole(nbh),
                                       )
        self.__dict__.update(self)


    #
    # miscellaneous methods
    #

    def __str__(self):
        fmt = self.__class__.__name__+'{'
        for key, obj in self.items():
            fmt += '\n{0},'.format(obj)
        fmt += '\n}'
        return fmt


    def __hash__(self):
        i = None
        for obj in self.values():
            if i is None:
                i = hash(obj)
            else:
                i ^= hash(obj)
        return i


    def __len__(self):
        return sum([obj.n for obj in self.values()])
    n = property(__len__)


    @property
    def empty(self):
        return self.__class__()


    def copy(self):
        return copy.deepcopy(self)


    def append(self, objs):
        if objs.n:
            if isinstance(objs, Particles):
                for (key, obj) in objs.items():
                    if obj.n:
                        self[key].append(obj)
            else:
                key = objs.__class__.__name__.lower()
                self[key].append(objs)


    def select(self, function):
        subset = self.empty
        for (key, obj) in self.items():
            if obj.n:
                subset[key].append(obj.select(function))
        return subset


    def stack_fields(self, attrs, pad=-1):
#        arrays = [obj.stack_fields(attrs, pad) for obj in self.values() if obj.n]
#        return np.concatenate(arrays)

        arrays = [(arr.reshape(-1,1) if arr.ndim < 2 else arr)
                  for arr in [getattr(self, attr) for attr in attrs]]
        array = np.concatenate(arrays, axis=1)

        ncols = array.shape[1]
        col = ncols - pad
        if col < 0:
            pad_array = np.zeros((len(array),pad), dtype=array.dtype)
            pad_array[:,:col] = array
            return pad_array
        if ncols > 1:
            return array
        return array.squeeze()


    #
    # common attributes
    #

    ### ...


    #
    # common methods
    #

    ### total mass and center-of-mass

    def get_total_mass(self):
        """
        Get the total mass.
        """
        return float(self.mass.sum())

    def get_center_of_mass_position(self):
        """
        Get the center-of-mass position.
        """
        mtot = self.get_total_mass()
        rcom = (self.mass * self.pos.T).sum(1)
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_center_of_mass_position"):
                    rcom += obj.get_pn_correction_for_center_of_mass_position()
        return (rcom / mtot)

    def get_center_of_mass_velocity(self):
        """
        Get the center-of-mass velocity.
        """
        mtot = self.get_total_mass()
        vcom = (self.mass * self.vel.T).sum(1)
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_center_of_mass_velocity"):
                    vcom += obj.get_pn_correction_for_center_of_mass_velocity()
        return (vcom / mtot)

    def correct_center_of_mass(self):
        """
        Correct the center-of-mass to origin of coordinates.
        """
        self.pos -= self.get_center_of_mass_position()
        self.vel -= self.get_center_of_mass_velocity()


    ### linear momentum

    def get_total_linear_momentum(self):
        """
        Get the total linear momentum.
        """
        lmom = (self.mass * self.vel.T).sum(1)
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_total_linear_momentum"):
                    lmom += obj.get_pn_correction_for_total_linear_momentum()
        return lmom


    ### angular momentum

    def get_total_angular_momentum(self):
        """
        Get the total angular momentum.
        """
        amom = (self.mass * np.cross(self.pos, self.vel).T).sum(1)
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_total_angular_momentum"):
                    amom += obj.get_pn_correction_for_total_angular_momentum()
        return amom


    ### kinetic energy

    def get_total_kinetic_energy(self):
        """
        Get the total kinetic energy.
        """
        ke = float((0.5 * self.mass * (self.vel**2).sum(1)).sum())
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_total_energy"):
                    ke += obj.get_pn_correction_for_total_energy()
        return ke


    ### potential energy

    def get_total_potential_energy(self):
        """
        Get the total potential energy.
        """
        return 0.5 * float((self.mass * self.phi).sum())


    ### update methods

    def update_nstep(self):
        """
        Update individual step number.
        """
        self.nstep += 1

    def update_t_curr(self, tau):
        """
        Update individual current time by tau.
        """
        self.t_curr += tau


    ### gravity

    def update_phi(self, objs):
        """
        Update the individual gravitational potential due to other particles.
        """
        gravity.phi.set_args(self, objs)
        gravity.phi.run()
        self.phi = gravity.phi.get_result()

    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other particles.
        """
        gravity.acc.set_args(self, objs)
        gravity.acc.run()
        self.acc = gravity.acc.get_result()

    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        gravity.tstep.set_args(self, objs, eta/2)
        gravity.tstep.run()
        self.dt_next = gravity.tstep.get_result()

    def update_acc_jerk(self, objs):
        """
        Update the individual gravitational acceleration and jerk due to other particles.
        """
        gravity.acc_jerk.set_args(self, objs)
        gravity.acc_jerk.run()
        (self.acc, jerk) = gravity.acc_jerk.get_result()
        for iobj in self.values():
            if iobj.n:
                iobj.jerk = jerk[:iobj.n]
                jerk = jerk[iobj.n:]

    def update_pnacc(self, objs, pn_order, clight):
        """
        Update the individual post-newtonian gravitational acceleration due to other particles.
        """
        ni = self.blackhole.n
        nj = objs.blackhole.n
        if ni and nj:
            gravity.pnacc.set_args(self.blackhole, objs.blackhole, pn_order, clight)
            gravity.pnacc.run()
            result = gravity.pnacc.get_result()
            for iobj in self.values():
                if iobj.n:
                    if hasattr(iobj, "pnacc"):
                        iobj.pnacc = result[:iobj.n]
                        result = result[iobj.n:]


    ### prev/next timestep

    def set_dt_prev(self, tau):
        """

        """
        for obj in self.values():
            if obj.n:
                obj.dt_prev = tau

    def set_dt_next(self, tau):
        """

        """
        for obj in self.values():
            if obj.n:
                obj.dt_next = tau


    def min_dt_prev(self):
        """

        """
        min_value = sys.float_info.max
        for obj in self.values():
            if obj.n:
                min_value = min(min_value, np.abs(obj.dt_prev).min())
        return min_value

    def min_dt_next(self):
        """

        """
        min_value = sys.float_info.max
        for obj in self.values():
            if obj.n:
                min_value = min(min_value, np.abs(obj.dt_next).min())
        return min_value


    def max_dt_prev(self):
        """

        """
        max_value = 0.0
        for obj in self.values():
            if obj.n:
                max_value = max(max_value, np.abs(obj.dt_prev).max())
        return max_value

    def max_dt_next(self):
        """

        """
        max_value = 0.0
        for obj in self.values():
            if obj.n:
                max_value = max(max_value, np.abs(obj.dt_next).max())
        return max_value


    def power_averaged_dt_prev(self, power):
        """

        """
        n = 0
        average = 0.0
        for obj in self.values():
            if obj.n:
                average += (obj.dt_prev**power).sum()
                n += obj.n
        average = (average / n)**(1.0/power)
        return average


    def power_averaged_dt_next(self, power):
        """

        """
        n = 0
        average = 0.0
        for obj in self.values():
            if obj.n:
                average += (obj.dt_next**power).sum()
                n += obj.n
        average = (average / n)**(1.0/power)
        return average


########## end of file ##########
